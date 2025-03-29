use lumine::models::bert::Model;
use std::env;
use std::error::Error;
use std::io::ErrorKind;
use std::path::Path;
use std::time::Instant;

/// Helper function for cosine similarity between two vectors
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() {
        return 0.0; // Or handle error
    }
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum();
    let norm1: f32 = v1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    dot_product / (norm1 * norm2)
}

/// Prints a subset of the embedding vector for display purposes
fn print_embedding_sample(name: &str, embedding: &[f32]) {
    let dims = embedding.len();
    let sample = if dims <= 10 {
        embedding.to_vec()
    } else {
        embedding[..10].to_vec()
    };

    println!("{} ({} dims): {:?}", name, dims, sample);
}

fn main() -> Result<(), Box<dyn Error>> {
    // Get model directory from command-line arguments or use default
    let args: Vec<String> = env::args().collect();
    let model_dir = if args.len() > 1 {
        Path::new(&args[1])
    } else {
        eprintln!("Usage: cargo run --example bert [path/to/model_directory]");
        eprintln!("No model directory provided, defaulting to './model_data'");
        Path::new("./model_data")
    };

    // Check if directory exists before loading
    if !model_dir.exists() {
        eprintln!("Model directory {:?} does not exist.", model_dir);
        eprintln!("Please download a suitable BERT model (e.g., from Hugging Face Hub)");
        eprintln!("and place config.json, tokenizer.json, model.safetensors in the directory.");
        return Err(Box::new(std::io::Error::new(
            ErrorKind::NotFound,
            "Model directory not found",
        )));
    }

    println!("Loading BERT model from {:?}...", model_dir);
    let start = Instant::now();
    let model = Model::from_path(model_dir)?;
    println!("Model loaded in {:.2?}", start.elapsed());

    // Example sentences to encode
    let sentences = [
        "This is an example sentence.",
        "Each sentence is converted to an embedding vector.",
        "Semantically similar sentences will have similar embeddings.",
        "The weather is nice today.",
        "The climate is pleasant this afternoon.",
    ];

    // Generate embeddings for all sentences
    println!(
        "\nGenerating embeddings for {} sentences...",
        sentences.len()
    );
    let start = Instant::now();
    let embeddings: Vec<Vec<f32>> = model
        .embedding(sentences.to_vec())
        .unwrap_or_else(|e| {
            eprintln!("Error generating embeddings: {}", e);
            vec![]
        })
        .into_iter()
        .map(|emb| emb.to_vec())
        .collect();
    println!("Embeddings generated in {:.2?}", start.elapsed());

    // Print info about the embeddings
    println!("\nEmbedding samples (first 10 dimensions):");
    for (i, embedding) in embeddings.iter().enumerate() {
        if embedding.is_empty() {
            println!("Embedding {} failed", i);
            continue;
        }
        print_embedding_sample(&format!("Embedding {}", i), embedding);
    }

    // Calculate and print similarity matrix
    println!("\nSimilarity matrix:");
    println!("{:-^50}", "");
    for i in 0..sentences.len() {
        for j in 0..sentences.len() {
            if embeddings[i].is_empty() || embeddings[j].is_empty() {
                print!("{:6.3} ", 0.0);
                continue;
            }
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            print!("{:6.3} ", sim);
        }
        println!();
    }
    println!("{:-^50}", "");

    // Print the original sentences for reference
    println!("\nSentences:");
    for (i, sentence) in sentences.iter().enumerate() {
        println!("[{}] {}", i, sentence);
    }

    // Example of semantic similarity demonstration
    println!("\nSemantic similarity examples:");
    // Similar sentences
    let sim_0_1 = if !embeddings[0].is_empty() && !embeddings[1].is_empty() {
        cosine_similarity(&embeddings[0], &embeddings[1])
    } else {
        0.0
    };
    println!("Similarity between [0] and [1]: {:.4}", sim_0_1);

    // Very similar sentences
    let sim_3_4 = if !embeddings[3].is_empty() && !embeddings[4].is_empty() {
        cosine_similarity(&embeddings[3], &embeddings[4])
    } else {
        0.0
    };
    println!("Similarity between [3] and [4]: {:.4}", sim_3_4);

    // Unrelated sentences
    let sim_0_3 = if !embeddings[0].is_empty() && !embeddings[3].is_empty() {
        cosine_similarity(&embeddings[0], &embeddings[3])
    } else {
        0.0
    };
    println!("Similarity between [0] and [3]: {:.4}", sim_0_3);

    Ok(())
}
