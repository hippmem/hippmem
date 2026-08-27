//! API backends: OpenAI-compatible clients for embedder and
//! extractor, plus Cohere (08 §3). Vendor-neutral: any service
//! implementing the OpenAI protocol works.

pub mod cohere;
pub mod openai;
pub mod openai_extract;
