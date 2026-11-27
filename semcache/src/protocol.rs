//! Wire protocol between the gateway and the sidecar.
//!
//! One JSON object per line in each direction, over a Unix socket or TCP.
//! The gateway keeps connections in the cosocket pool, so one connection
//! carries many requests, one at a time.
//!
//! ```text
//! -> {"op":"lookup","id":"<request id>","ns":"<namespace>","text":"...","threshold":0.9 (optional)}
//! <- {"ok":true,"hit":true,"score":0.93,"response":{...},"embed_ms":6.1}
//!    with the cross-encoder check also "rerank":0.98,"rerank_ms":11.2
//! -> {"op":"store","id":"<request id>","ns":"<namespace>","text":"...","ttl":3600,"response":{...}}
//! <- {"ok":true,"stored":true}
//! -> {"op":"route","id":"<request id>","text":"..."}
//! <- {"ok":true,"score":0.64,"strong":true,"route_ms":6.3,"embed_ms":6.1,"version":"3f2a..."}
//! -> {"op":"guard","text":"..."}
//! <- {"ok":true,"score":0.98,"injection":true,"guard_ms":7.9,"version":"9c1d..."}
//! -> {"op":"stats"}
//! -> {"op":"ping"}
//! <- {"ok":false,"error":"..."}
//! ```

use crate::index::CachedResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Lookup {
        /// Request id. The sidecar keeps the query embedding under it for a
        /// short time, so a store after a miss does not embed the text again.
        id: String,
        ns: String,
        text: String,
        /// Optional override. Without it, the sidecar uses the threshold that
        /// belongs to its model (see `--threshold` and semcache.json). With the
        /// cross-encoder check, it is the cosine floor for a candidate.
        threshold: Option<f32>,
    },
    Store {
        id: String,
        ns: String,
        text: String,
        /// Time to live in seconds.
        ttl: u64,
        response: CachedResponse,
    },
    /// Score the prompt with the trained router (--router). A head router
    /// keeps the embedding under the request id for a short time, so the
    /// cache lookup that follows does not embed the same text again.
    Route {
        id: Option<String>,
        text: String,
    },
    /// Score the prompt with the guardrail classifier (--guard).
    Guard {
        text: String,
    },
    Stats,
    Ping,
}

#[derive(Debug, Default, Serialize)]
pub struct Reply {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    // lookup
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hit: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<CachedResponse>,
    /// Set when the sidecar refuses to cache this text, for example "too_long".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_ms: Option<f64>,
    /// Best cross-encoder probability among the candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_ms: Option<f64>,

    // store
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored: Option<bool>,

    // route (the router score is in `score`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strong: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_ms: Option<f64>,

    // guard (the classifier score is in `score`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub injection: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard_ms: Option<f64>,

    /// Version of the model that answered a route or guard request
    /// (semcache::version).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    // stats
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<serde_json::Value>,
}

impl Reply {
    pub fn ok() -> Self {
        Self { ok: true, ..Default::default() }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self { ok: false, error: Some(message.into()), ..Default::default() }
    }
}
