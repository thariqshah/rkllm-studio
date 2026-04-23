mod openai;
mod rkllm;

use axum::{
    extract::{State, Json},
    response::{sse::{Event, Sse}, IntoResponse},
    routing::{get, post},
    Router,
};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::Utc;
use tower_http::cors::CorsLayer;

use crate::openai::{ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Usage, ModelList, Model};
use crate::rkllm::RKLLMEngine;

struct AppState {
    engine: Mutex<Option<RKLLMEngine>>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState {
        engine: Mutex::new(None),
    });

    let app = Router::new()
        .route("/", get(|| async { "RKLLama API Server is running!" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/load", post(load_model))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8181").await.unwrap();
    tracing::info!("RKLLama server running on http://0.0.0.0:8181");
    axum::serve(listener, app).await.unwrap();
}

async fn list_models() -> Json<ModelList> {
    let mut models = Vec::new();
    
    let paths = vec!["/ai-shit/docker-runtime", "models", "../models"];
    for path_str in paths {
        if let Ok(entries) = std::fs::read_dir(path_str) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("rkllm") {
                    if let Some(file_name) = path.file_stem().and_then(|s| s.to_str()) {
                        models.push(Model {
                            id: file_name.to_string(),
                            object: "model".to_string(),
                            created: chrono::Utc::now().timestamp(),
                            owned_by: "rkllama".to_string(),
                        });
                    }
                }
            }
            if !models.is_empty() { break; }
        }
    }

    if models.is_empty() {
        models.push(Model {
            id: "placeholder-no-models".to_string(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "rkllama".to_string(),
        });
    }

    Json(ModelList {
        object: "list".to_string(),
        data: models,
    })
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_name = payload.get("path").and_then(|v| v.as_str()).unwrap_or("gemma.rkllm");
    
    // Check multiple paths for the model
    let possible_paths = vec![
        format!("/ai-shit/docker-runtime/{}", model_name),
        format!("models/{}", model_name),
        format!("../models/{}", model_name),
        model_name.to_string(), // Direct path
    ];

    let mut final_path = None;
    for p in possible_paths {
        if std::path::Path::new(&p).exists() {
            final_path = Some(p);
            break;
        }
    }

    let model_path = match final_path {
        Some(p) => {
            // Convert to absolute path for the C library
            match std::fs::canonicalize(&p) {
                Ok(abs_path) => abs_path.to_string_lossy().to_string(),
                Err(_) => p,
            }
        },
        None => return (axum::http::StatusCode::NOT_FOUND, "Model file not found").into_response(),
    };
    
    tracing::info!("Loading model from absolute path: {}", model_path);
    
    let engine = match RKLLMEngine::new(&model_path) {
        Ok(e) => e,
        Err(e) => return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load model: {}", e)).into_response(),
    };
    let mut current_engine = state.engine.lock().await;
    *current_engine = Some(engine);

    Json(serde_json::Value::String("Model loaded successfully".to_string())).into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let engine_opt = state.engine.lock().await;
    let engine = match &*engine_opt {
        Some(e) => e,
        None => return (axum::http::StatusCode::BAD_REQUEST, "Model not loaded").into_response(),
    };

    let prompt = request.messages.last().map(|m| m.content.clone()).unwrap_or_default();
    
    if request.stream.unwrap_or(false) {
        let (tx, mut rx) = tokio::sync::mpsc::channel(100);
        
        // We need to keep the engine lock or use an Arc. 
        // For RKLLM, we can only have one active inference or handle it via a queue.
        // For now, we'll run it in the background.
        
        // Since we can't easily clone the engine (it's not Clone), 
        // we'll use the one from state directly if we can, or just run it here.
        
        // In a real multi-user scenario, you'd want a queue.
        // For now, we assume the user is the only one.
        
        let prompt = prompt.clone();
        
        // We'll drop the lock after starting run to allow other non-inference requests (like stats)
        // Note: rkllm_run is async at the library level if param.is_async = true.
        let res = engine.run(&prompt, tx).await;
        if let Err(e) = res {
             return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
        }

        let model_name = request.model.clone();
        let stream = async_stream::stream! {
            while let Some(text) = rx.recv().await {
                if text == "\n[DONE]" { break; }
                let chunk = crate::openai::ChatCompletionChunk {
                    id: format!("chatcmpl-{}", Uuid::new_v4()),
                    object: "chat.completion.chunk".to_string(),
                    created: Utc::now().timestamp(),
                    model: model_name.clone(),
                    choices: vec![crate::openai::ChunkChoice {
                        index: 0,
                        delta: crate::openai::Delta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            }
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(stream).into_response()
    } else {
        // Non-streaming response (simplified mock)
        Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: request.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: "This is a mock response from RKLLama.".to_string(),
                    name: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
        }).into_response()
    }
}
