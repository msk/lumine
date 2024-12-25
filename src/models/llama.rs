use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use minijinja::context;
use serde::Serialize;
use tokenizers::Tokenizer;

const CHAT_TEMPLATE_NAME: &str = "chat";

/// A language model that can be used for various NLP tasks.
pub struct Model {
    device: Device,
    tokenizer: Tokenizer,
    weights: ModelWeights,
    eos_token: u32,

    /// The template environment for the chat template, stored with the name
    /// `CHAT_TEMPLATE_NAME`.
    template_env: minijinja::Environment<'static>,
}

impl Model {
    /// Loads a model in the GGUF format from the specified path.
    ///
    /// # Errors
    ///
    /// This method returns an error if the model cannot be loaded from the
    /// specified path.
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};

        let device = crate::device();

        let mut file = std::fs::File::open(&path)?;
        let gguf = gguf_file::Content::read(&mut file)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e.with_path(path)))?;
        let tokenizer_json = gguf
            .metadata
            .get("tokenizer.huggingface.json")
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "missing tokenizer.huggingface.json in metadata",
                )
            })?
            .to_string()
            .map_err(|_| {
                Error::new(
                    ErrorKind::InvalidData,
                    "tokenizer.huggingface.json is not a string",
                )
            })?;
        let tokenizer: Tokenizer = tokenizer_json
            .parse()
            .map_err(|e| Error::new(ErrorKind::InvalidData, format!("invalid tokenizer: {e}")))?;
        let eos_token = gguf
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "missing tokenizer.ggml.eos_token_id in metadata",
                )
            })?
            .to_u32()
            .map_err(|_| {
                Error::new(
                    ErrorKind::InvalidData,
                    "tokenizer.ggml.eos_token_id is not a valid u32",
                )
            })?;
        let chat_template = gguf
            .metadata
            .get("tokenizer.chat_template")
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "missing tokenizer.ggml.chat_template in metadata",
                )
            })?
            .to_string()
            .map_err(|_| {
                Error::new(
                    ErrorKind::InvalidData,
                    "tokenizer.ggml.chat_template is not a string",
                )
            })?;
        let mut template_env = minijinja::Environment::new();
        template_env
            .add_template_owned(CHAT_TEMPLATE_NAME.to_string(), chat_template.to_string())
            .map_err(|e| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("invalid chat template: {e}"),
                )
            })?;

        let weights = ModelWeights::from_gguf(gguf, &mut file, &device).map_err(|e| {
            Error::new(
                ErrorKind::InvalidData,
                format!("invalid model weights: {e}"),
            )
        })?;

        Ok(Self {
            device,
            tokenizer,
            weights,
            eos_token,
            template_env,
        })
    }

    /// Creates an iterator that generates tokens that completes the chat
    /// starting with the given prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt cannot be tokenized. This can happen if
    /// the model's tokenizer is unable to encode the prompt or the chat
    /// template is invalid.
    pub fn completions(&mut self, prompt: &str) -> std::io::Result<Completions> {
        use std::io::{Error, ErrorKind};

        #[derive(Serialize)]
        struct ChatMessage<'role, 'content> {
            role: &'role str,
            content: &'content str,
        }

        let Ok(chat_template) = self.template_env.get_template(CHAT_TEMPLATE_NAME) else {
            unreachable!("template `CHAT_TEMPLATE_NAME` always exists")
        };
        let messages = vec![ChatMessage {
            role: "user",
            content: prompt,
        }];
        let prompt = chat_template
            .render(context!(messages => messages, add_generation_prompt => true))
            .map_err(|e| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("chat template doesn't conform to the expected format: {e}"),
                )
            })?;

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, e))?
            .get_ids()
            .to_vec();
        let logits_processor = {
            let sampling = Sampling::All { temperature: 0.8 };
            LogitsProcessor::from_sampling(299_792_458, sampling)
        };

        let last_output_pos = tokens.len();
        Ok(Completions {
            device: &self.device,
            tokenizer: &self.tokenizer,
            weights: &mut self.weights,
            eos_token: self.eos_token,
            logits_processor,
            offset: 0,
            tokens,
            input_pos: 0,
            last_output_pos,
        })
    }
}

/// An iterator that generates tokens to complete a chat.
pub struct Completions<'a> {
    device: &'a Device,
    tokenizer: &'a Tokenizer,
    weights: &'a mut ModelWeights,
    eos_token: u32,
    logits_processor: LogitsProcessor,
    offset: usize,
    tokens: Vec<u32>,
    input_pos: usize,
    last_output_pos: usize,
}

impl Iterator for Completions<'_> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.tokens.is_empty() {
                break;
            }

            let input = Tensor::new(&self.tokens[self.input_pos..], self.device)
                .ok()?
                .unsqueeze(0)
                .ok()?;
            let logits = self
                .weights
                .forward(&input, self.offset)
                .ok()?
                .squeeze(0)
                .ok()?;
            let next_token = self.logits_processor.sample(&logits).ok()?;
            if next_token == self.eos_token {
                let prev_text = self
                    .tokenizer
                    .decode(&self.tokens[..self.last_output_pos], true)
                    .expect("should be valid tokens");
                let curr_text = self
                    .tokenizer
                    .decode(&self.tokens, true)
                    .expect("should be valid tokens");
                let rest = curr_text.split_at(prev_text.len()).1;
                self.tokens.clear();
                return Some(rest.to_string());
            }
            self.offset += if self.offset == 0 {
                self.tokens.len()
            } else {
                1
            };

            let prev_text = if self.last_output_pos == 0 {
                String::new()
            } else {
                self.tokenizer
                    .decode(&self.tokens[..self.last_output_pos], true)
                    .expect("should be valid tokens")
            };
            self.tokens.push(next_token);
            let curr_text = self
                .tokenizer
                .decode(&self.tokens, true)
                .expect("should be valid tokens");
            if curr_text.len() > prev_text.len()
                && curr_text
                    .chars()
                    .last()
                    .expect("not empty")
                    .is_alphanumeric()
            {
                let text = curr_text.split_at(prev_text.len());
                self.input_pos = 0;
                self.last_output_pos = 1;
                self.tokens.clear();
                self.tokens.push(next_token);
                return Some(text.1.to_string());
            }
            self.input_pos += 1;
        }
        None
    }
}
