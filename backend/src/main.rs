use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Router,
};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tower_http::cors::CorsLayer;
use futures::stream::StreamExt;
use std::path::{Path, PathBuf};

// Use the prelude for correct imports from rkllm-rs
use rkllm_rs::prelude::*;

mod openai;
use openai::{Model, ModelList, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage, Usage};

struct AppState {
    // LLMHandle does not implement Clone, so we must wrap it in an Arc to share it across threads
    engine: Mutex<Option<Arc<LLMHandle>>>,
}

// Handler that bridges the RKLLM callback to a Tokio mpsc channel
struct StreamHandler {
    tx: mpsc::Sender<String>,
}

impl RkllmCallbackHandler for StreamHandler {
    fn handle(&mut self, result: Option<RKLLMResult<'_>>, state: LLMCallState) {
        if let Some(res) = result {
            let text = res.text.as_ref();
            if !text.is_empty() {
                let _ = self.tx.try_send(text.to_string());
            }
        }
        
        match state {
            LLMCallState::Finish => {
                let _ = self.tx.try_send("[DONE]".to_string());
            }
            LLMCallState::Error => {
                let _ = self.tx.try_send("[ERROR]".to_string());
            }
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState {
        engine: Mutex::new(None),
    });

    let app = Router::new()
        .route("/", get(|| async { "RKLLama API Server is running (using rkllm-rs)!" }))
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
    }

    let mut seen = std::collections::HashSet::new();
    models.retain(|m| seen.insert(m.id.clone()));

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
        Some(p) => std::fs::canonicalize(&p).unwrap_or(p),
        None => return (StatusCode::NOT_FOUND, "Model file not found").into_response(),
    };
    
    let model_path = model_path_buf.to_string_lossy().to_string();
    tracing::info!("Loading model using rkllm-rs: {}", model_path);
    
    let mut config = LLMConfig::default();
    config.model_path = Some(model_path);
    config.max_context_len = 2048;
    config.max_new_tokens = 512;
    config.top_k = 40;
    config.top_p = 0.9;
    config.temperature = 0.8;
    
    let handle = match rkllm_rs::prelude::init(config) {
        Ok(h) => h,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to initialize RKLLM: {}", e)).into_response(),
    };
    
    let mut current_engine = state.engine.lock().await;
    // Wrap the handle in an Arc so we can share it across requests
    *current_engine = Some(Arc::new(handle));

    Json(serde_json::Value::String("Model loaded successfully".to_string())).into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Clone the Arc while holding the lock. This is cheap and ensures the handle
    // stays alive independently of the state's lock.
    let engine_arc = {
        let guard = state.engine.lock().await;
        match guard.as_ref() {
            Some(e) => Arc::clone(e),
            None => return (StatusCode::BAD_REQUEST, "No model loaded").into_response(),
        }
    };

    let prompt = req.messages.last().map(|m| m.content.clone()).unwrap_or_default();
    let (tx, rx) = mpsc::channel(1024);
    let handler = StreamHandler { tx };

    // Move the cloned Arc into the task. Since Arc<LLMHandle> is 'static,
    // this satisfies the spawn_blocking requirement.
    tokio::task::spawn_blocking(move || {
        // Use the recommended prompt constructor
        let input = RKLLMInput::prompt(prompt);
        // Arc<T> implements Deref<Target=T>, so we can call run() directly
        if let Err(e) = engine_arc.run(input, None, handler) {
            tracing::error!("Inference error: {}", e);
        }
    });

    if req.stream.unwrap_or(false) {
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
            .map(|text| {
                if text == "[DONE]" || text == "[ERROR]" {
                    return None;
                }

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
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                Some(Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap())))
            })
            .filter_map(|x| async move { x });

        Sse::new(stream).into_response()
    } else {
        (StatusCode::NOT_IMPLEMENTED, "Non-streaming not implemented yet").into_response()
    }
}
