use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Router,
};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use futures::stream::Stream;
use std::path::{Path, PathBuf};

mod rkllm;
mod openai;

use rkllm::{RKLLMEngine, RKLLMResult};
use openai::{Model, ModelList, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage, Usage};

struct AppState {
    engine: Mutex<Option<RKLLMEngine>>,
}

#[tokio::main]
async fn main() {
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

fn find_rkllm_files(dir: &Path, models: &mut Vec<Model>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_rkllm_files(&path, models);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rkllm") {
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
    }
}

async fn list_models() -> Json<ModelList> {
    let mut models = Vec::new();
    let paths = vec!["/ai-shit/docker-runtime", "models", "../models"];
    
    for path_str in paths {
        find_rkllm_files(Path::new(path_str), &mut models);
        if !models.is_empty() { break; }
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

fn find_specific_model(dir: &Path, name: &str) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_specific_model(&path, name) {
                    return Some(found);
                }
            } else {
                let stem = path.file_stem().and_then(|s| s.to_str());
                let full_name = path.file_name().and_then(|s| s.to_str());
                if stem == Some(name) || full_name == Some(name) {
                    return Some(path);
                }
            }
        }
    }
    None
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_name = payload.get("path").and_then(|v| v.as_str()).unwrap_or("gemma.rkllm");
    
    let search_paths = vec!["/ai-shit/docker-runtime", "models", "../models"];
    let mut final_path = None;

    for p in search_paths {
        if let Some(found) = find_specific_model(Path::new(p), model_name) {
            final_path = Some(found);
            break;
        }
    }

    let model_path_buf = match final_path {
        Some(p) => {
            match std::fs::canonicalize(&p) {
                Ok(abs_path) => abs_path,
                Err(_) => p,
            }
        },
        None => return (StatusCode::NOT_FOUND, "Model file not found").into_response(),
    };
    
    let model_path = model_path_buf.to_string_lossy().to_string();
    tracing::info!("Loading model from absolute path: {}", model_path);
    
    let engine = match RKLLMEngine::new(&model_path) {
        Ok(e) => e,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to load model: {}", e)).into_response(),
    };
    
    let mut current_engine = state.engine.lock().await;
    *current_engine = Some(engine);

    Json(serde_json::Value::String("Model loaded successfully".to_string())).into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let engine_opt = state.engine.lock().await;
    let engine = match engine_opt.as_ref() {
        Some(e) => e,
        None => return (StatusCode::BAD_REQUEST, "No model loaded").into_response(),
    };

    let prompt = req.messages.last().map(|m| m.content.clone()).unwrap_or_default();
    let rx = engine.run(&prompt);

    if req.stream.unwrap_or(false) {
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
            .map(|text| {
                let chunk = ChatCompletionResponse {
                    id: "chat-123".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: "gemma-3-4b".to_string(),
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()))
            });

        Sse::new(stream).into_response()
    } else {
        // Simple non-streaming fallback (buffer all tokens)
        // ... omitted for brevity as we focus on streaming
        (StatusCode::NOT_IMPLEMENTED, "Non-streaming not implemented yet").into_response()
    }
}
