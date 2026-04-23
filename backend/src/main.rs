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
use std::ffi::{CString, c_void};
use std::ptr;

// Use rkllm-sys for raw FFI control, but rkllm-rs for the handle wrapper if possible
use rkllm_sys_rs::{
    rkllm_init, rkllm_run, rkllm_destroy, rkllm_result, rkllm_call_state,
    rkllm_input, rkllm_input_type, rkllm_param, rkllm_extend_param, rkllm_infer_param,
};

mod openai;
use openai::{Model, ModelList, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage, Usage};

// Thread-safe wrapper for the raw handle
struct SafeLLMHandle {
    handle: *mut c_void,
}

unsafe impl Send for SafeLLMHandle {}
unsafe impl Sync for SafeLLMHandle {}

impl Drop for SafeLLMHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { rkllm_destroy(self.handle); }
        }
    }
}

struct AppState {
    engine: Mutex<Option<Arc<SafeLLMHandle>>>,
}

// Data passed to the C callback
struct RequestCtx {
    tx: mpsc::Sender<String>,
}

unsafe extern "C" fn rkllm_callback(
    result: *mut rkllm_result,
    userdata: *mut c_void,
    state: rkllm_call_state
) {
    if userdata.is_null() { return; }
    let ctx = &*(userdata as *const RequestCtx);

    if !result.is_null() {
        let res = &*result;
        if !res.text.is_null() {
            let text = std::ffi::CStr::from_ptr(res.text).to_string_lossy().into_owned();
            if !text.is_empty() {
                let _ = ctx.tx.try_send(text);
            }
        }
    }

    match state {
        rkllm_call_state::RKLLM_RUN_FINISH => {
            let _ = ctx.tx.try_send("[DONE]".to_string());
        }
        rkllm_call_state::RKLLM_RUN_ERROR => {
            let _ = ctx.tx.try_send("[ERROR]".to_string());
        }
        _ => {}
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let state = Arc::new(AppState {
        engine: Mutex::new(None),
    });

    let app = Router::new()
        .route("/", get(|| async { "RKLLama API Server is running (Native FFI)!" }))
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
    tracing::info!("Initializing RKLLM with raw FFI: {}", model_path);
    
    let c_model_path = CString::new(model_path).unwrap();
    
    unsafe {
        let mut param: rkllm_param = std::mem::zeroed();
        param.model_path = c_model_path.as_ptr() as *mut i8;
        param.max_context_len = 2048;
        param.max_new_tokens = 512;
        param.top_k = 40;
        param.top_p = 0.9;
        param.temperature = 0.8;
        
        // Correct alignment and defaults for RK3588
        let mut extend_param: rkllm_extend_param = std::mem::zeroed();
        extend_param.base_domain_id = 1;
        // Big cores mask: (1<<4 | 1<<5 | 1<<6 | 1<<7) = 240
        extend_param.enabled_cpus_mask = 240; 
        param.extend_param = &mut extend_param;
        
        let mut handle: *mut c_void = ptr::null_mut();
        let ret = rkllm_init(&mut handle, &mut param, Some(rkllm_callback));
        
        if ret != 0 {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("rkllm_init failed with code {}", ret)).into_response();
        }
        
        let mut current_engine = state.engine.lock().await;
        *current_engine = Some(Arc::new(SafeLLMHandle { handle }));
    }

    Json(serde_json::Value::String("Model loaded successfully".to_string())).into_response()
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let engine = {
        let guard = state.engine.lock().await;
        match guard.as_ref() {
            Some(e) => Arc::clone(e),
            None => return (StatusCode::BAD_REQUEST, "No model loaded").into_response(),
        }
    };

    let prompt_str = req.messages.last().map(|m| m.content.clone()).unwrap_or_default();
    let (tx, rx) = mpsc::channel(1024);
    
    tokio::task::spawn_blocking(move || {
        let c_prompt = CString::new(prompt_str).unwrap();
        let ctx = RequestCtx { tx };
        
        unsafe {
            let mut input: rkllm_input = std::mem::zeroed();
            input.type_ = rkllm_input_type::RKLLM_INPUT_PROMPT;
            input.__bindgen_anon_1.prompt = c_prompt.as_ptr() as *mut i8;
            
            let mut infer_param: rkllm_infer_param = std::mem::zeroed();
            
            // This blocks until rkllm_callback sends RKLLM_RUN_FINISH
            let ret = rkllm_run(engine.handle, &mut input, &mut infer_param, &ctx as *const _ as *mut c_void);
            if ret != 0 {
                tracing::error!("rkllm_run failed with code {}", ret);
            }
            
            // Ensure c_prompt and ctx stay alive until rkllm_run returns
            drop(c_prompt);
            drop(ctx);
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
                    model: "rkllm".to_string(),
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
