# wasmedge-llmc
A Rust library for using llm.c functions when the Wasi is being executed on WasmEdge.

## Set up WasmEdge
```bash
git clone https://github.com/WasmEdge/WasmEdge.git
cd WasmEdge
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_BUILD_TESTS=OFF -DWASMEDGE_PLUGIN_LLMC=ON
cmake --build build
cmake --install build
```

## Download Checkpoints & Training data
```bash
wget -P /tmp/data/ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_124M.bin
wget -P /tmp/data/ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_train.bin
wget -P /tmp/data/ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_val.bin
wget -P /tmp/data/ https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/gpt2_tokenizer.bin
```

## Run the example
```bash
cd example
cargo build --target wasm32-wasi --release
wasmedge --dir .:. ./target/wasm32-wasi/release/example.wasm
```
