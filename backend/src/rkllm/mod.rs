use libc::{c_char, c_void, c_int};
use std::ffi::{CStr, CString};
use std::ptr;
use tokio::sync::mpsc;

// Opaque handle to the LLM instance
pub type LLMHandle = *mut c_void;

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
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
    pub last_hidden_layer: *mut c_void,
    pub logits: *mut c_void,
    pub perf_info: RKLLMPerfInfo,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMPerfInfo {
    pub prefill_tokens: i32,
    pub prefill_time: f32,
    pub decode_tokens: i32,
    pub decode_time: f32,
}

pub type LLMResultCallback = unsafe extern "C" fn(result: *mut RKLLMResult, userdata: *mut c_void, state: LLMCallState);

#[repr(C)]
pub struct RKLLMParam {
    pub model_path: *const c_char,
    pub max_context_len: i32,
    pub max_new_tokens: i32,
    pub top_k: i32,
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
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMExtendParam {
    pub base_npu_core: i32,
    pub reserved: [u8; 104],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RKLLMInput {
    pub input_type: RKLLMInputType,
    pub input: RKLLMInputUnion,
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum RKLLMInputType {
    RKLLM_INPUT_PROMPT = 0,
    RKLLM_INPUT_TOKEN = 1,
    RKLLM_INPUT_EMBED = 2,
    RKLLM_INPUT_MULTIMODAL = 3,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union RKLLMInputUnion {
    pub prompt: *const c_char,
    pub tokens: RKLLMTokens,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMTokens {
    pub tokens: *mut i32,
    pub n_tokens: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RKLLMInferParam {
    pub mode: i32,
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
}

unsafe extern "C" fn rkllm_callback_wrapper(result: *mut RKLLMResult, userdata: *mut c_void, state: LLMCallState) {
    if userdata.is_null() || result.is_null() { return; }
    
    let ctx = &*(userdata as *mut CallbackCtx);
    let res = &*result;
    
    if !res.text.is_null() {
        let c_str = CStr::from_ptr(res.text);
        if let Ok(s) = c_str.to_str() {
            let _ = ctx.tx.try_send(s.to_string());
        }
    }

    match state {
        LLMCallState::RKLLM_RUN_FINISH => {
            let _ = ctx.tx.try_send("\n[DONE]".to_string());
            let _ = Box::from_raw(userdata as *mut CallbackCtx);
        }
        LLMCallState::RKLLM_RUN_ERROR => {
            let _ = ctx.tx.try_send("\n[ERROR]".to_string());
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
            param.top_k = 40;
            param.top_p = 0.9;
            param.temperature = 0.8;
            param.repeat_penalty = 1.1;
            param.is_async = true;

            let ret = rkllm_init(&mut handle, &mut param, rkllm_callback_wrapper);
            if ret != 0 {
                return Err(format!("Failed to initialize RKLLM: {}", ret));
            }

            Ok(RKLLMEngine { handle })
        }
    }

    pub async fn run(&self, prompt: &str, tx: mpsc::Sender<String>) -> Result<(), String> {
        let ctx = Box::new(CallbackCtx { tx });
        let userdata = Box::into_raw(ctx) as *mut c_void;

        unsafe {
            let c_prompt = CString::new(prompt).map_err(|_| "Invalid prompt")?;
            let mut input = RKLLMInput {
                input_type: RKLLMInputType::RKLLM_INPUT_PROMPT,
                input: RKLLMInputUnion { prompt: c_prompt.as_ptr() },
            };

            let mut infer_param = std::mem::zeroed::<RKLLMInferParam>();
            infer_param.mode = 0;

            let ret = rkllm_run(self.handle, &mut input, &mut infer_param, userdata);
            if ret != 0 {
                let _ = Box::from_raw(userdata as *mut CallbackCtx);
                return Err(format!("Failed to run RKLLM: {}", ret));
            }
        }
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
