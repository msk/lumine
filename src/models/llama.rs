use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use minijinja::context;
use serde::Serialize;
use tokenizers::Tokenizer;

const CHAT_TEMPLATE_NAME: &str = "chat";

/// A chat message with a role and content.
#[derive(Serialize)]
pub struct ChatMessage<'role, 'content> {
    role: &'role str,
    content: &'content str,
}

impl<'role, 'content> ChatMessage<'role, 'content> {
    /// Creates a new chat message with the specified role and content.
    #[must_use]
    pub fn new(role: &'role str, content: &'content str) -> Self {
        Self { role, content }
    }
}

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
    pub fn completions(&mut self, messages: &[ChatMessage]) -> std::io::Result<Completions> {
        use std::io::{Error, ErrorKind};

        let Ok(chat_template) = self.template_env.get_template(CHAT_TEMPLATE_NAME) else {
            unreachable!("template `CHAT_TEMPLATE_NAME` always exists")
        };
        let prompt = prefilled_prompt(&chat_template, messages)?;

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

    /// Returns the probabilities of the given tokens being the next token in the
    /// completion of the chat starting with the given prompt.
    ///
    /// The n-th element in the returned vector corresponds to the probability
    /// of the n-th token in the given `tokens` being the next token in the LLM
    /// completion.
    ///
    /// # Errors
    ///
    /// Returns an error if `messages` contain invalid tokens.
    pub fn next_probabilities(
        &mut self,
        messages: &[ChatMessage],
        tokens: &[&str],
    ) -> std::io::Result<Vec<f32>> {
        use std::io::{Error, ErrorKind};

        let Ok(chat_template) = self.template_env.get_template(CHAT_TEMPLATE_NAME) else {
            unreachable!("template `CHAT_TEMPLATE_NAME` always exists")
        };
        let prompt = prefilled_prompt(&chat_template, messages)?;

        let tokens = {
            let mut token_ids = Vec::with_capacity(tokens.len());
            for &token in tokens {
                let token_id = *self.tokenizer.get_vocab(true).get(token).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidInput, format!("invalid token: {token}"))
                })?;
                token_ids.push(token_id);
            }
            token_ids
        };

        // Encode the question using the tokenizer
        let tokenized_prompt = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, e))?
            .get_ids()
            .to_vec();
        if tokenized_prompt.is_empty() {
            return Err(Error::new(ErrorKind::InvalidInput, "empty question"));
        }

        let input = Tensor::new(tokenized_prompt, &self.device)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid tokens: {e}")))?
            .unsqueeze(0)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid question: {e}")))?;
        let logits = self
            .weights
            .forward(&input, 0)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid question: {e}")))?
            .squeeze(0)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid question: {e}")))?;
        let logits = logits
            .to_dtype(DType::F32)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid question: {e}")))?;
        let prs = candle_nn::ops::softmax_last_dim(&logits)
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid question: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| Error::new(ErrorKind::InvalidInput, format!("invalid question: {e}")))?;

        // Calculate the probability of the answer being "yes"
        let mut token_prs = Vec::with_capacity(tokens.len());
        for token in tokens {
            let p = *prs
                .get(token as usize)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "token not in vocab"))?;
            token_prs.push(p);
        }
        Ok(token_prs)
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
            self.input_pos = self.tokens.len();
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
        }
        None
    }
}

/// Returns a prompt with the response prefilled if the last message's role is
/// "assistant".
///
/// # Errors
///
/// Returns an error if the chat template doesn't conform to the expected format.
fn prefilled_prompt(
    template: &minijinja::Template,
    messages: &[ChatMessage],
) -> std::io::Result<String> {
    use std::io::{Error, ErrorKind};

    let Some(last_message) = messages.last() else {
        return Ok(String::new());
    };
    if last_message.role == "assistant" {
        let messages = &messages[..messages.len() - 1];
        template
            .render(context!(messages => messages, add_generation_prompt => true))
            .map(|s| s + last_message.content)
            .map_err(|e| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("chat template doesn't conform to the expected format: {e}"),
                )
            })
    } else {
        template
            .render(context!(messages => messages, add_generation_prompt => true))
            .map_err(|e| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("chat template doesn't conform to the expected format: {e}"),
                )
            })
    }
}
