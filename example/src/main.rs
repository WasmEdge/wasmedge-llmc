use wasmedge_llmc::*;

fn main() {
    let config = ConfigBuilder::default().lr(0.0002).epoch(20).build();
    let model = match Model::from_checkpoints("/tmp/data/gpt2_124M.bin") {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {:?}", e);
            return;
        }
    };
    let train_dataloader = match DataLoader::from_file(
        "/tmp/data/tiny_shakespeare_train.bin",
        4,    // batch size
        64,   // sequence length
        0,    // process rank
        1,    // number of processes
        true, // should shuffle
    ) {
        Ok(loader) => loader,
        Err(e) => {
            eprintln!("Failed to load training data: {:?}", e);
            return;
        }
    };
    let val_dataloader = match DataLoader::from_file(
        "/tmp/data/tiny_shakespeare_val.bin",
        4,     // batch size
        64,    // sequence length
        0,     // process rank
        1,     // number of processes
        false, // should shuffle
    ) {
        Ok(loader) => loader,
        Err(e) => {
            eprintln!("Failed to load validation data: {:?}", e);
            return;
        }
    };
    let tokenizer = match Tokenizer::from_file("/tmp/data/gpt2_tokenizer.bin") {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {:?}", e);
            return;
        }
    };
    match model.train(train_dataloader, val_dataloader, tokenizer, config) {
        Ok(_) => println!("Training completed successfully."),
        Err(e) => eprintln!("Failed to train model: {:?}", e),
    }
}
