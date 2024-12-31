# Lumine

Lumine is a high-level Rust interface for language models powered by the
[Candle](https://github.com/huggingface/candle) ML framework. It provides
ergonomic APIs for tasks such as text generation and other natural language
processing (NLP) applications. By leveraging Candle's efficient Rust
implementation, Lumine ensures safe and intuitive language model inference
without compromising on performance.

## Features

- High-level API for language model inference
- Support for text generation and other NLP tasks
- Efficient implementation using the Candle ML framework
- CUDA and Metal support for accelerated inference

## Notes

- Lumine currently supports only text generation.
- The "llama" architecture is the only supported model architecture. There are
  many open-weight models available with this architecture, including SmolLM2.
  More architectures will be added in the future.
- The GGUF file must contain the Hugging Face tokenizer in its metadata, under
  the key `tokenizer.huggingface.json`.
- The SmolLM2 model containing the Hugging Face tokenizer can be downloaded from
  the Hugging Face website:
  [SmolLM2-HFT-1.7B-Instruct](https://huggingface.co/minskim/SmolLM2-HFT-1.7B-Instruct).
- The API is not stable until it reaches version 1.0 and has not been released
  to crates.io yet. The best way to handle the model architecture and tokenizer
  in a more generic way is still being explored.

## Usage

Add the following to your `Cargo.toml`:

```toml
[dependencies]
lumine = { git = "https://github.com/msk/lumine", branch = "main" }
```

To enable CUDA or Metal support, use the corresponding feature flags:

```toml
[dependencies]
lumine = { git = "https://github.com/msk/lumine", branch = "main", features = ["cuda"] }
```

or

```toml
[dependencies]
lumine = { git = "https://github.com/msk/lumine", branch = "main", features = ["metal"] }
```

Here's a basic example of how to use Lumine to load a model and generate text
completions:

```rust
use lumine::models::llama::{ChatMessage, Model};

fn main() -> std::io::Result<()> {
    // Load the model from a GGUF file
    let mut model = Model::from_path("path/to/model.gguf")?;

    // Generate a completion for a given prompt
    let messages = vec![
        ChatMessage::new("system", "You are a helpful assistant."),
        ChatMessage::new("user", "What is the capital of South Korea?"),
    ];
    let mut completions = model.completions(&messages)?;

    // Print the generated text
    while let Some(text) = completions.next() {
        print!("{}", text);
    }
    println!();

    Ok(())
}
```

## License

Licensed under either of

- Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
