use super::*;
use axum::{
    extract::{MatchedPath, Request},
    middleware::Next,
};
use std::{sync::Mutex, time::Instant};

#[derive(Default)]
pub(super) struct Metrics {
    requests: Mutex<BTreeMap<(String, String, u16), (u64, f64)>>,
}
pub(super) async fn record(
    State(metrics): State<Arc<Metrics>>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().to_string();
    let path = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str())
        .unwrap_or("unmatched")
        .to_owned();
    let start = Instant::now();
    let response = next.run(request).await;
    let mut entries = metrics.requests.lock().unwrap_or_else(|e| e.into_inner());
    let entry = entries
        .entry((method, path, response.status().as_u16()))
        .or_default();
    entry.0 += 1;
    entry.1 += start.elapsed().as_secs_f64();
    response
}
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
pub(super) async fn endpoint(State(state): State<ServerState>) -> Response {
    let entries = state
        .metrics
        .requests
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut output = String::from(
        "# HELP ruda_http_requests_total HTTP responses started.\n# TYPE ruda_http_requests_total counter\n# HELP ruda_http_response_headers_seconds Time until HTTP response headers, not full SSE lifetime.\n# TYPE ruda_http_response_headers_seconds summary\n",
    );
    for ((method, path, status), (count, seconds)) in entries.iter() {
        let labels = format!(
            "method=\"{}\",path=\"{}\",status=\"{status}\"",
            label(method),
            label(path)
        );
        output.push_str(&format!("ruda_http_requests_total{{{labels}}} {count}\nruda_http_response_headers_seconds_count{{{labels}}} {count}\nruda_http_response_headers_seconds_sum{{{labels}}} {seconds}\n"));
    }
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        output,
    )
        .into_response()
}
