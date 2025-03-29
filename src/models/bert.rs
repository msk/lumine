use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::fs;
use std::io::{self, Error as IoError, ErrorKind};
use std::path::Path;
use tokenizers::{PaddingParams, Tokenizer};

/// The BERT embedding model.
pub struct Model {
    device: Device,
    tokenizer: Tokenizer,
    weights: BertModel,
}

impl Model {
    /// Creates a new BERT model from the given directory.
    ///
    /// The directory must contain the following files:
    /// - config.json: The model configuration file.
    /// - model.safetensors: The model weights file (safetensors format expected).
    /// - tokenizer.json: The tokenizer file.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the files is missing or invalid.
    pub fn from_path<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::load_model(path.as_ref())
    }

    // Internal function using io::Result for error handling
    fn load_model(path: &Path) -> io::Result<Self> {
        let device = crate::device();

        // Check files to load
        let config_path = path.join("config.json");
        if !config_path.exists() {
            return Err(IoError::new(
                ErrorKind::NotFound,
                format!("Config file not found: {}", config_path.display()),
            ));
        }
        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(IoError::new(
                ErrorKind::NotFound,
                format!("Tokenizer file not found: {}", tokenizer_path.display()),
            ));
        }
        let weights_path = path.join("model.safetensors");
        if !weights_path.exists() {
            return Err(IoError::new(
                ErrorKind::NotFound,
                format!("Weights file not found: {}", weights_path.display()),
            ));
        }

        let config_str = fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str).map_err(|e| {
            IoError::new(
                ErrorKind::InvalidData,
                format!(
                    "Failed to parse config file: {}: {e}",
                    config_path.display()
                ),
            )
        })?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            IoError::new(
                ErrorKind::Other,
                format!(
                    "Failed to load tokenizer: {}: {e}",
                    tokenizer_path.display()
                ),
            )
        })?;
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device).map_err(|e| {
                IoError::new(ErrorKind::Other, format!("Failed to load weights: {e}"))
            })?
        };
        let weights = BertModel::load(vb, &config).map_err(|e| {
            IoError::new(
                ErrorKind::Other,
                format!("Failed to load BertModel from VarBuilder and config: {e}"),
            )
        })?;

        println!("Model loaded successfully."); // Optional: Logging
        Ok(Self {
            device,
            tokenizer,
            weights,
        })
    }

    /// Returns the embedding for the given text.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or model inference fails.
    pub fn embedding(&self, input: Vec<&str>) -> io::Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(input, true) // `true` adds special tokens
            .map_err(|e| {
                IoError::new(ErrorKind::InvalidInput, format!("Tokenization failed: {e}"))
            })?;
        let tokens_ids = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_ids(), &self.device))
            .collect::<Result<Vec<_>, candle_core::Error>>()
            .map_err(|e| {
                IoError::new(
                    ErrorKind::Other,
                    format!("Failed to create token_ids tensor: {e}"),
                )
            })?;
        let attention_mask = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_attention_mask(), &self.device))
            .collect::<Result<Vec<_>, candle_core::Error>>()
            .map_err(|e| {
                IoError::new(
                    ErrorKind::Other,
                    format!("Failed to create attention_mask tensor: {e}"),
                )
            })?;

        let stacked_token_ids = Tensor::stack(&tokens_ids, 0).map_err(|e| {
            IoError::new(
                ErrorKind::Other,
                format!("Failed to stack token_ids tensor: {e}"),
            )
        })?;
        let attention_mask = Tensor::stack(&attention_mask, 0).map_err(|e| {
            IoError::new(
                ErrorKind::Other,
                format!("Failed to stack attention_mask tensor: {e}"),
            )
        })?;

        let token_type_ids = stacked_token_ids.zeros_like().map_err(|e| {
            IoError::new(
                ErrorKind::Other,
                format!("Failed to create token_type_ids tensor: {e}"),
            )
        })?;

        let model_output = self
            .weights
            .forward(&stacked_token_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| {
                IoError::new(ErrorKind::Other, format!("Model forward pass failed: {e}"))
            })?;
        let embeddings = model_output.sum(1).map_err(|e| {
            IoError::new(ErrorKind::Other, format!("Failed to sum model output: {e}"))
        })?;

        let mut final_embedding_vec = Vec::with_capacity(encodings.len());
        for i in 0..encodings.len() {
            let Ok(embedding_i) = embeddings.get(i) else {
                unreachable!("a valid index");
            };
            let embedding = embedding_i.to_vec1().map_err(|e| {
                IoError::new(
                    ErrorKind::Other,
                    format!("Failed to convert embedding to Vec<f32>: {e}"),
                )
            })?;
            final_embedding_vec.push(embedding);
        }
        Ok(final_embedding_vec)
    }
}
