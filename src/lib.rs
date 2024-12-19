//! Lumine is a high-level interface for language models powered by the Candle
//! ML framework. It provides ergonomic APIs for tasks such as text generation
//! and other natural language processing (NLP) applications. By leveraging
//! Candle's efficient Rust implementation, Lumine ensures safe and intuitive
//! language model inference without compromising on performance.

pub mod models;

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device,
};

fn device() -> Device {
    if cuda_is_available() {
        candle_core::cuda::set_gemm_reduced_precision_f16(true);
        candle_core::cuda::set_gemm_reduced_precision_bf16(true);
        if let Ok(device) = Device::new_cuda(0) {
            return device;
        }
    }

    if metal_is_available() {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
    }

    Device::Cpu
}
