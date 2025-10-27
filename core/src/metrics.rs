use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f32,
    pub accuracy: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub loss: f32,
    pub accuracy: f32,
}
