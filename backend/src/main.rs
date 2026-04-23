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
    Json(ModelList {
        object: "list".to_string(),
        data: vec![Model {
            id: "llama-3-8b".to_string(),
            object: "model".to_string(),
            created: 1715000000,
            owned_by: "rkllama".to_string(),
        }],
    })
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_path = payload.get("path").and_then(|v| v.as_str()).unwrap_or("models/gemma.rkllm");
    
    let engine = RKLLMEngine::new(model_path).unwrap();
    let mut current_engine = state.engine.lock().await;
    *current_engine = Some(engine);

    Json(serde_json::Value::String("Model loaded successfully".to_string()))
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
