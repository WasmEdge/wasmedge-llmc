use core::fmt;
use std::error::Error;
#[repr(transparent)]
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct WasmedgeLLMCErrno(u32);
pub const WASMEDGE_LLM_ERRNO_SUCCESS: WasmedgeLLMCErrno = WasmedgeLLMCErrno(0);
pub const WASMEDGE_LLM_ERRNO_INVALID_ARGUMENT: WasmedgeLLMCErrno = WasmedgeLLMCErrno(1);
pub const WASMEDGE_LLM_ERRNO_MISSING_MEMORY: WasmedgeLLMCErrno = WasmedgeLLMCErrno(2);
impl WasmedgeLLMCErrno {
    pub const fn raw(&self) -> u32 {
        self.0
    }

    pub fn name(&self) -> &'static str {
        match self.0 {
            0 => "SUCCESS",
            1 => "INVALID_ARGUMENT",
            2 => "MISSING_MEMORY",
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }
    pub fn message(&self) -> &'static str {
        match self.0 {
            0 => "",
            1 => "",
            2 => "",
            3 => "",
            4 => "",
            5 => "",
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }
}
impl fmt::Debug for WasmedgeLLMCErrno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmedgeLLMCErrno")
            .field("code", &self.0)
            .field("name", &self.name())
            .field("message", &self.message())
            .finish()
    }
}
impl fmt::Display for WasmedgeLLMCErrno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (error {})", self.name(), self.0)
    }
}

impl Error for WasmedgeLLMCErrno {}

#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
impl std::error::Error for WasmedgeLLMCErrno {}

pub unsafe fn model_create(
    checkpoint_path: &str,
    model_id: *mut u32,
) -> Result<(), WasmedgeLLMCErrno> {
    let checkpoint_path_ptr = checkpoint_path.as_ptr() as u32;
    let checkpoint_path_len = checkpoint_path.len() as u32;
    let model_id_ptr = model_id as u32;
    let result = wasmedge_llm::model_create(checkpoint_path_ptr, checkpoint_path_len, model_id_ptr);
    if result != 0 {
        Err(WasmedgeLLMCErrno(result as u32))
    } else {
        Ok(())
    }
}

pub unsafe fn dataloader_create(
    data_path: &str,
    batch_size: u32,
    sequence_length: u32,
    process_rank: u32,
    num_processes: u32,
    should_shuffule: bool,
    dataloader_id: *mut u32,
) -> Result<(), WasmedgeLLMCErrno> {
    let data_path_str = data_path.as_ptr() as u32;
    let data_path_len = data_path.len() as u32;
    let dataloader_id_ptr = dataloader_id as u32;
    let result = wasmedge_llm::dataloader_create(
        data_path_str,
        data_path_len,
        batch_size,
        sequence_length,
        process_rank,
        num_processes,
        should_shuffule as u32,
        dataloader_id_ptr,
    );
    if result != 0 {
        Err(WasmedgeLLMCErrno(result as u32))
    } else {
        Ok(())
    }
}

pub unsafe fn tokenizer_create(
    filepath: &str,
    tokenizer_id: *mut u32,
) -> Result<(), WasmedgeLLMCErrno> {
    let filepath_ptr = filepath.as_ptr() as u32;
    let filepath_len = filepath.len() as u32;
    let tokenizer_id_ptr = tokenizer_id as u32;
    let result = wasmedge_llm::tokenizer_create(filepath_ptr, filepath_len, tokenizer_id_ptr);
    if result != 0 {
        Err(WasmedgeLLMCErrno(result as u32))
    } else {
        Ok(())
    }
}

pub unsafe fn model_train(
    model_id: u32,
    train_dataloader_id: u32,
    val_dataloader_id: u32,
    tokenizer_id: u32,
    batch_size: u32,
    sequence_length: u32,
    lr: f32,
    epoch: u32,
) -> Result<(), WasmedgeLLMCErrno> {
    let result = wasmedge_llm::model_train(
        model_id,
        train_dataloader_id,
        val_dataloader_id,
        tokenizer_id,
        batch_size,
        sequence_length,
        lr,
        epoch,
    );

    if result != 0 {
        Err(WasmedgeLLMCErrno(result as u32))
    } else {
        Ok(())
    }
}

pub mod wasmedge_llm {
    #[link(wasm_import_module = "wasmedge_llmc")]
    extern "C" {
        pub fn model_create(
            checkpoint_path_ptr: u32,
            checkpoint_path_len: u32,
            model_id_ptr: u32,
        ) -> i32;

        pub fn dataloader_create(
            data_path_str: u32,
            data_path_len: u32,
            batch_size: u32,
            sequence_length: u32,
            process_rank: u32,
            num_processes: u32,
            should_shuffule: u32,
            dataloader_id_ptr: u32,
        ) -> i32;

        pub fn tokenizer_create(filepath_ptr: u32, filepath_len: u32, tokenizer_id: u32) -> i32;

        pub fn model_train(
            model_id: u32,
            train_dataloader_id: u32,
            val_dataloader_id: u32,
            tokenizer_id: u32,
            batch_size: u32,
            sequence_length: u32,
            lr: f32,
            epoch: u32,
        ) -> i32;
    }
}
