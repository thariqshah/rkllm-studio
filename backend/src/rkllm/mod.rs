use libc::{c_char, c_void, c_int, c_float, size_t};
use std::ffi::{CStr, CString};
use std::ptr;
use tokio::sync::mpsc;

// Opaque handle to the LLM instance
pub type LLMHandle = *mut c_void;

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LLMCallState {
    RKLLM_RUN_NORMAL = 0,
    RKLLM_RUN_WAITING = 1,
    RKLLM_RUN_FINISH = 2,
    RKLLM_RUN_ERROR = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMResult {
    pub text: *const c_char,
    pub token_id: i32,
    pub last_hidden_layer: RKLLMResultLastHiddenLayer,
    pub logits: RKLLMResultLogits,
    pub perf: RKLLMPerfStat,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMResultLastHiddenLayer {
    pub hidden_states: *mut f32,
    pub embd_size: i32,
    pub num_tokens: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMResultLogits {
    pub logits: *mut f32,
    pub vocab_size: i32,
    pub num_tokens: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMPerfStat {
    pub prefill_time_ms: f32,
    pub prefill_tokens: i32,
    pub generate_time_ms: f32,
    pub generate_tokens: i32,
    pub memory_usage_mb: f32,
}

pub type LLMResultCallback = unsafe extern "C" fn(result: *mut RKLLMResult, userdata: *mut c_void, state: LLMCallState);

#[repr(C)]
pub struct RKLLMParam {
    pub model_path: *const c_char,
    pub max_context_len: i32,
    pub max_new_tokens: i32,
    pub top_k: f32,
    pub n_keep: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub mirostat: i32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub skip_special_token: bool,
    pub is_async: bool,
    pub img_start: *const c_char,
    pub img_end: *const c_char,
    pub img_content: *const c_char,
    pub extend_param: RKLLMExtendParam,
    pub use_gpu: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMExtendParam {
    pub base_domain_id: i32,
    pub embed_flash: i8,
    pub enabled_cpus_num: i8,
    pub enabled_cpus_mask: u32,
    pub n_batch: u8,
    pub use_cross_attn: i8,
    pub reserved: [u8; 104],
}

pub const RKLLM_INPUT_PROMPT: i32 = 0;
pub const RKLLM_INPUT_TOKEN: i32 = 1;
pub const RKLLM_INPUT_EMBED: i32 = 2;
pub const RKLLM_INPUT_MULTIMODAL: i32 = 3;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RKLLMInput {
    pub role: *const c_char,
    pub enable_thinking: bool,
    pub input_type: i32,
    pub input: RKLLMInputUnion,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union RKLLMInputUnion {
    pub prompt: *const c_char,
    pub embed: RKLLMEmbedInput,
    pub token: RKLLMTokenInput,
    pub multimodal: RKLLMMultiModelInput,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMEmbedInput {
    pub embed: *mut f32,
    pub n_tokens: size_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMTokenInput {
    pub input_ids: *mut i32,
    pub n_tokens: size_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMMultiModelInput {
    pub prompt: *const c_char,
    pub image_embed: *mut f32,
    pub n_image_tokens: size_t,
    pub n_image: size_t,
    pub image_width: size_t,
    pub image_height: size_t,
}

#[repr(C)]
pub struct RKLLMInferParam {
    pub mode: i32,
    pub lora_params: *mut c_void,
    pub prompt_cache_params: *mut c_void,
    pub keep_history: i32,
}

#[link(name = "rkllmrt")]
extern "C" {
    pub fn rkllm_createDefaultParam() -> RKLLMParam;
    pub fn rkllm_init(handle: *mut LLMHandle, param: *mut RKLLMParam, callback: LLMResultCallback) -> c_int;
    pub fn rkllm_run(handle: LLMHandle, input: *mut RKLLMInput, infer_param: *mut RKLLMInferParam, userdata: *mut c_void) -> c_int;
    pub fn rkllm_destroy(handle: LLMHandle) -> c_int;
}

#[derive(Clone)]
pub struct RKLLMEngine {
    handle: LLMHandle,
}

struct CallbackCtx {
    tx: mpsc::Sender<String>,
    done_tx: mpsc::Sender<()>,
    _prompt: CString,
}

unsafe extern "C" fn rkllm_callback_wrapper(result: *mut RKLLMResult, userdata: *mut c_void, state: LLMCallState) {
    if userdata.is_null() || result.is_null() { return; }
    
    let ctx = &mut *(userdata as *mut CallbackCtx);
    let res = &*result;
    
    if !res.text.is_null() {
        if let Ok(s) = CStr::from_ptr(res.text).to_str() {
            let _ = ctx.tx.try_send(s.to_string());
        }
    }

    match state {
        LLMCallState::RKLLM_RUN_FINISH | LLMCallState::RKLLM_RUN_ERROR => {
            if state == LLMCallState::RKLLM_RUN_FINISH {
                let _ = ctx.tx.try_send("\n[DONE]".to_string());
            }
            let _ = ctx.done_tx.try_send(());
            let _ = Box::from_raw(userdata as *mut CallbackCtx);
        }
        _ => {}
    }
}

impl RKLLMEngine {
    pub fn new(model_path: &str) -> Result<Self, String> {
        unsafe {
            let mut handle: LLMHandle = ptr::null_mut();
            let mut param = rkllm_createDefaultParam();
            
            let c_model_path = CString::new(model_path).map_err(|_| "Invalid path")?;
            param.model_path = c_model_path.as_ptr();
            param.max_context_len = 2048;
            param.max_new_tokens = 512;
            param.top_k = 40.0;
            param.top_p = 0.9;
            param.temperature = 0.8;
            param.repeat_penalty = 1.1;
            param.is_async = true;
            param.use_gpu = true;

            // Affinity for RK3588 Big Cores
            param.extend_param.enabled_cpus_num = 4;
            param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7);

            let ret = rkllm_init(&mut handle, &mut param, rkllm_callback_wrapper);
            if ret != 0 {
                return Err(format!("Failed to initialize RKLLM: {}", ret));
            }

            Ok(RKLLMEngine { handle })
        }
    }

    pub async fn run(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<(), String> {
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let c_prompt = CString::new(prompt).map_err(|_| "Invalid prompt")?;
        
        unsafe {
            let ctx = Box::new(CallbackCtx { 
                tx, 
                done_tx, 
                _prompt: c_prompt 
            });
            let userdata = Box::into_raw(ctx) as *mut c_void;
            
            let persistent_prompt_ptr = (*(userdata as *mut CallbackCtx))._prompt.as_ptr();
            
            let mut input = RKLLMInput {
                role: ptr::null(),
                enable_thinking: false,
                input_type: RKLLM_INPUT_PROMPT,
                input: RKLLMInputUnion { prompt: persistent_prompt_ptr },
            };
            
            let mut infer_param = std::mem::zeroed::<RKLLMInferParam>();
            infer_param.mode = 0;
            
            let ret = rkllm_run(self.handle, &mut input, &mut infer_param, userdata);
            if ret != 0 {
                let _ = Box::from_raw(userdata as *mut CallbackCtx);
                return Err(format!("RKLLM run failed: {}", ret));
            }
        }
        
        let _ = done_rx.recv().await;
        Ok(())
    }
}

impl Drop for RKLLMEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                rkllm_destroy(self.handle);
            }
        }
    }
}

unsafe impl Send for RKLLMEngine {}
unsafe impl Sync for RKLLMEngine {}
