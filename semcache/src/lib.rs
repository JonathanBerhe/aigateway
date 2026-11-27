//! Semantic cache for aigateway.
//!
//! The gateway sends the prompt of a cacheable request to this sidecar. The
//! sidecar embeds it with MiniLM (Burn) and searches a vector index for a
//! cached answer to a similar prompt. A similar prompt is one whose cosine
//! similarity reaches the threshold that the gateway sends.
//!
//! Why a sidecar and not the FFI library: an embedding takes milliseconds,
//! and an FFI call blocks the NGINX worker for that time. A sidecar also
//! holds one model and one index for all NGINX workers.

pub mod classifier;
pub mod embed;
pub mod guard;
pub mod index;
pub mod protocol;
pub mod rerank;
pub mod router;
pub mod routing;
pub mod server;
pub mod version;
