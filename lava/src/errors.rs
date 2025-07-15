#[derive(Debug, thiserror::Error)]
#[error("Missing {0}")]
pub struct SuitablityError(pub &'static str);
