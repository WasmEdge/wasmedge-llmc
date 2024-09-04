pub mod llmc_interface;
use core::mem::MaybeUninit;
use llmc_interface::*;

pub struct Model {
    id: u32,
}

pub struct Config {
    lr: f32,
    epoch: u32,
}

pub struct ConfigBuilder {
    lr: f32,
    epoch: u32,
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self {
            lr: 0.0001,
            epoch: 20,
        }
    }
}

impl ConfigBuilder {
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn epoch(mut self, epoch: u32) -> Self {
        self.epoch = epoch;
        self
    }

    pub fn build(self) -> Config {
        Config {
            lr: self.lr,
            epoch: self.epoch,
        }
    }
}

impl Model {
    pub fn from_checkpoints(checkpoint_path: &str) -> Result<Self, WasmedgeLLMCErrno> {
        let mut model_id = MaybeUninit::<u32>::uninit();
        unsafe {
            let result = model_create(checkpoint_path, model_id.as_mut_ptr());
            if let Err(code) = result {
                return Err(code);
            }
            Ok(Model {
                id: model_id.assume_init(),
            })
        }
    }

    pub fn train(
        &self,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        tokenizer: Tokenizer,
        config: Config,
    ) -> Result<(), WasmedgeLLMCErrno> {
        unsafe {
            model_train(
                self.id,
                train_data_loader.id,
                val_data_loader.id,
                tokenizer.id,
                train_data_loader.batch_size,
                train_data_loader.sequence_length,
                config.lr,
                config.epoch,
            )
        }
    }
}

pub struct DataLoader {
    id: u32,
    batch_size: u32,
    sequence_length: u32,
}

impl DataLoader {
    pub fn from_file(
        data_path: &str,
        batch_size: u32,
        sequence_length: u32,
        process_rank: u32,
        num_processes: u32,
        should_shuffle: bool,
    ) -> Result<Self, WasmedgeLLMCErrno> {
        let mut dataloader_id = MaybeUninit::<u32>::uninit();
        unsafe {
            let result = dataloader_create(
                data_path,
                batch_size,
                sequence_length,
                process_rank,
                num_processes,
                should_shuffle,
                dataloader_id.as_mut_ptr(),
            );
            if let Err(code) = result {
                return Err(code);
            }
            Ok(DataLoader {
                id: dataloader_id.assume_init(),
                batch_size,
                sequence_length,
            })
        }
    }
}

pub struct Tokenizer {
    id: u32,
}

impl Tokenizer {
    pub fn from_file(filepath: &str) -> Result<Self, WasmedgeLLMCErrno> {
        let mut tokenizer_id = MaybeUninit::<u32>::uninit();
        unsafe {
            let result = tokenizer_create(filepath, tokenizer_id.as_mut_ptr());
            if let Err(code) = result {
                return Err(code);
            }
            Ok(Tokenizer {
                id: tokenizer_id.assume_init(),
            })
        }
    }
}
