use std::{
    env,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use burn::{
    module::Param,
    nn::{
        loss::{CrossEntropyLoss, CrossEntropyLossConfig},
        Linear,
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::ElementConversion,
    tensor::{
        activation::relu,
        backend::{AutodiffBackend, Backend},
        Int, Tensor, TensorData,
    },
};
use burn_autodiff::Autodiff;
use burn_candle::{Candle, CandleDevice};
use burn_dataset::{vision::MnistDataset, Dataset};
use gildnn_core::{
    encode_luma_png_data_url, ensure_report_file, load_or_init, seeded_rng, update_sections,
    EvaluationMetrics, ReportSection, StepMetrics, DEFAULT_REPORT_TEMPLATE,
};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};

const INPUT_DIM: usize = 28 * 28;
const NUM_CLASSES: usize = 10;
const HIDDEN_DIM: usize = 128;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 1e-3;
const TEST_FIVE_SHOT_BATCHES: usize = 5;
const SAMPLE_COUNT: usize = 3;
const FIVE_SHOT_STEP: usize = 10;
const FULL_TRAIN_STEPS: usize = 200;
const TEST_TRAIN_STEPS: usize = 25;
const BENCHMARK_TOLERANCE: f32 = 5e-3;

type TrainingBackend = Autodiff<Candle<f32, i64>>;

enum RunMode {
    Full,
    Test,
}

impl RunMode {
    fn parse() -> Result<Self> {
        let mut args = env::args().skip(1);
        let mut mode: Option<Self> = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--mode" | "-m" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("expected value after {}", arg))?;
                    mode = Some(Self::from_str(&value)?);
                }
                s if s.starts_with("--mode=") => {
                    let value = s.split_once('=').unwrap().1;
                    mode = Some(Self::from_str(value)?);
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => {
                    return Err(anyhow!("unexpected argument: {}", arg));
                }
            }
        }

        Ok(mode.unwrap_or(Self::Full))
    }

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "full" => Ok(Self::Full),
            "test" => Ok(Self::Test),
            other => Err(anyhow!("invalid mode: {}", other)),
        }
    }

    fn train_steps(&self) -> usize {
        match self {
            Self::Full => FULL_TRAIN_STEPS,
            Self::Test => TEST_TRAIN_STEPS,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Test => "test",
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ExperimentConfig {
    seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchmarkSnapshot {
    final_train: StepMetrics,
    final_test: EvaluationMetrics,
    five_shot: Option<EvaluationMetrics>,
}

struct ExperimentPaths {
    config: PathBuf,
    report: PathBuf,
    benchmark: PathBuf,
}

struct ExperimentResult {
    history: Vec<StepMetrics>,
    five_shot: Option<EvaluationMetrics>,
    final_test: EvaluationMetrics,
    final_train: StepMetrics,
    train_samples: Vec<SamplePrediction>,
    test_samples: Vec<SamplePrediction>,
}

#[derive(Clone)]
struct SamplePrediction {
    index: usize,
    label: usize,
    prediction: usize,
    image_data_url: String,
}

#[derive(Clone)]
struct MnistBatch<B: AutodiffBackend> {
    images: Tensor<B, 2>,
    labels: Tensor<B, 1, Int>,
}

#[derive(burn::module::Module, Debug)]
struct MnistClassifier<B: burn::tensor::backend::Backend> {
    hidden: Linear<B>,
    output: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> MnistClassifier<B> {
    fn init(device: &B::Device, seed: u64) -> Self {
        let mut rng = seeded_rng(seed);
        let hidden = linear_from_rng::<B>(&mut rng, device, INPUT_DIM, HIDDEN_DIM);
        let output = linear_from_rng::<B>(&mut rng, device, HIDDEN_DIM, NUM_CLASSES);

        Self { hidden, output }
    }

    fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = relu(self.hidden.forward(inputs));
        self.output.forward(hidden)
    }
}

fn main() -> Result<()> {
    let mode = RunMode::parse()?;
    let paths = initialize_paths()?;
    let config: ExperimentConfig = load_or_init(&paths.config, || ExperimentConfig { seed: 1337 })?;
    ensure_report_file(&paths.report, DEFAULT_REPORT_TEMPLATE)?;
    let benchmark = load_benchmark(&paths.benchmark)?;

    println!("running MNIST baseline in {} mode", mode.label());

    let result = run_training(mode.train_steps(), &config)?;

    write_report(&paths.report, &config, mode.train_steps(), &result)?;

    match mode {
        RunMode::Full => {
            if benchmark.is_none() {
                println!(
                    "no benchmark snapshot recorded yet; run with --mode test to capture one."
                );
            }
        }
        RunMode::Test => {
            let snapshot = BenchmarkSnapshot {
                final_train: result.final_train.clone(),
                final_test: result.final_test,
                five_shot: result.five_shot,
            };
            if let Some(reference) = benchmark {
                validate_benchmark(&snapshot, &reference)?;
                println!(
                    "benchmark check passed (tolerance {:.1e})",
                    BENCHMARK_TOLERANCE
                );
            } else {
                save_benchmark(&paths.benchmark, &snapshot)?;
                println!(
                    "saved new benchmark snapshot to {}",
                    paths.benchmark.display()
                );
            }
        }
    }

    Ok(())
}

fn print_usage() {
    println!("Usage: cargo run -p gildnn-experiment-mnist -- [--mode full|test]");
}

fn initialize_paths() -> Result<ExperimentPaths> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("experiments/mnist_baseline");
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create experiment directory {}", dir.display()))?;

    Ok(ExperimentPaths {
        config: dir.join("config.json"),
        report: dir.join("report.md"),
        benchmark: dir.join("benchmark.json"),
    })
}

fn run_training(train_steps: usize, config: &ExperimentConfig) -> Result<ExperimentResult> {
    let device = CandleDevice::Cpu;
    let mut model: MnistClassifier<TrainingBackend> = MnistClassifier::init(&device, config.seed);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    let train_dataset = MnistDataset::train();
    let test_dataset = MnistDataset::test();

    let mut history = Vec::with_capacity(train_steps);
    let mut five_shot: Option<EvaluationMetrics> = None;

    for step in 0..train_steps {
        let batch = training_batch::<TrainingBackend>(&train_dataset, &device, step);
        let logits = model.forward(batch.images.clone());
        let loss = loss_fn.forward(logits.clone(), batch.labels.clone());
        let (correct, total) = accuracy_counts(logits, batch.labels.clone());
        let accuracy = correct as f32 / total as f32 * 100.0;
        let loss_scalar = loss.clone().into_scalar().elem::<f32>();

        println!(
            "step {:03}: loss {:.4}, train accuracy {:.2}%",
            step + 1,
            loss_scalar,
            accuracy
        );

        history.push(StepMetrics {
            step: step + 1,
            loss: loss_scalar,
            accuracy,
        });

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(LEARNING_RATE, model, grads);

        if step + 1 == FIVE_SHOT_STEP {
            five_shot = Some(evaluate_dataset(
                &model,
                &loss_fn,
                &test_dataset,
                &device,
                Some(TEST_FIVE_SHOT_BATCHES),
            ));
            if let Some(eval) = five_shot {
                println!(
                    "step {:03}: five-shot test loss {:.4}, accuracy {:.2}%",
                    step + 1,
                    eval.loss,
                    eval.accuracy
                );
            }
        }
    }

    let final_eval = evaluate_dataset(&model, &loss_fn, &test_dataset, &device, None);

    println!(
        "final test: loss {:.4}, accuracy {:.2}%",
        final_eval.loss, final_eval.accuracy
    );

    let final_train = history
        .last()
        .cloned()
        .ok_or_else(|| anyhow!("training history is empty"))?;

    let train_samples = sample_predictions(&model, &device, &train_dataset, SAMPLE_COUNT)?;
    let test_samples = sample_predictions(&model, &device, &test_dataset, SAMPLE_COUNT)?;

    Ok(ExperimentResult {
        history,
        five_shot,
        final_test: final_eval,
        final_train,
        train_samples,
        test_samples,
    })
}

fn write_report(
    report_path: &Path,
    config: &ExperimentConfig,
    train_steps: usize,
    result: &ExperimentResult,
) -> Result<()> {
    let sections = [
        ReportSection::new(
            "configuration",
            render_configuration_section(config, train_steps),
        ),
        ReportSection::new(
            "metrics",
            render_metrics_section(&result.history, result.five_shot, &result.final_test),
        ),
        ReportSection::new(
            "samples-primary",
            render_samples_section("Training split", &result.train_samples),
        ),
        ReportSection::new(
            "samples-secondary",
            render_samples_section("Test split", &result.test_samples),
        ),
    ];

    update_sections(report_path, &sections)
}

fn render_configuration_section(config: &ExperimentConfig, train_steps: usize) -> String {
    format!(
        "- Seed: {}\n- Batch size: {}\n- Hidden units: {}\n- Train steps: {}\n- Learning rate: {:.4}\n- Five-shot evaluation batches: {}\n",
        config.seed, BATCH_SIZE, HIDDEN_DIM, train_steps, LEARNING_RATE, TEST_FIVE_SHOT_BATCHES
    )
}

fn render_metrics_section(
    history: &[StepMetrics],
    five_shot: Option<EvaluationMetrics>,
    final_eval: &EvaluationMetrics,
) -> String {
    let mut output = String::new();

    if let Some(eval) = five_shot {
        let _ = writeln!(
            &mut output,
            "- Step {} five-shot loss: {:.4}\n- Step {} five-shot accuracy: {:.2}%",
            FIVE_SHOT_STEP, eval.loss, FIVE_SHOT_STEP, eval.accuracy
        );
    }

    let final_train = history.last();
    if let Some(train) = final_train {
        let _ = writeln!(
            &mut output,
            "- Final train loss: {:.4}\n- Final train accuracy: {:.2}%",
            train.loss, train.accuracy
        );
    }

    let _ = writeln!(
        &mut output,
        "- Final test loss: {:.4}\n- Final test accuracy: {:.2}%\n",
        final_eval.loss, final_eval.accuracy
    );

    if !history.is_empty() {
        let _ = writeln!(&mut output, "| Step | Train Loss | Train Accuracy (%) |");
        let _ = writeln!(&mut output, "| --- | --- | --- |");

        for metrics in summarize_history(history) {
            let _ = writeln!(
                &mut output,
                "| {} | {:.4} | {:.2} |",
                metrics.step, metrics.loss, metrics.accuracy
            );
        }
    }

    output
}

fn render_samples_section(title: &str, samples: &[SamplePrediction]) -> String {
    if samples.is_empty() {
        return format!("### {}\n\nNo samples available for this split.", title);
    }

    let mut output = String::new();
    let _ = writeln!(&mut output, "### {}\n", title);

    for (i, sample) in samples.iter().enumerate() {
        let _ = writeln!(
            &mut output,
            "#### Sample {} (index {})\n- True label: {}\n- Predicted: {}\n\n![Sample image]({})\n",
            i + 1,
            sample.index,
            sample.label,
            sample.prediction,
            sample.image_data_url
        );
    }

    output
}

fn summarize_history<'a>(history: &'a [StepMetrics]) -> Vec<&'a StepMetrics> {
    if history.is_empty() {
        return Vec::new();
    }

    let total_steps = history.last().unwrap().step;
    let mut checkpoints = vec![1, 5, 10, 25, 50, 100, 150, total_steps];
    checkpoints.retain(|&step| step <= total_steps);
    checkpoints.sort_unstable();
    checkpoints.dedup();

    let mut summary = Vec::new();
    for target in checkpoints {
        if let Some(metrics) = history.iter().find(|m| m.step == target) {
            summary.push(metrics);
        }
    }

    if summary.last().map(|metrics| metrics.step) != history.last().map(|metrics| metrics.step) {
        summary.push(history.last().unwrap());
    }

    summary
}

fn training_batch<B: AutodiffBackend>(
    dataset: &MnistDataset,
    device: &B::Device,
    step: usize,
) -> MnistBatch<B> {
    let len = dataset.len();
    let start = (step * BATCH_SIZE) % len;
    let items: Vec<_> = (0..BATCH_SIZE)
        .map(|offset| dataset.get((start + offset) % len).unwrap())
        .collect();

    MnistBatch::from_items(device, &items)
}

fn evaluate_dataset(
    model: &MnistClassifier<TrainingBackend>,
    loss_fn: &CrossEntropyLoss<TrainingBackend>,
    dataset: &MnistDataset,
    device: &CandleDevice,
    max_batches: Option<usize>,
) -> EvaluationMetrics {
    let mut total_loss = 0.0;
    let mut total_correct = 0usize;
    let mut total_samples = 0usize;
    let mut batches = 0usize;
    let mut index = 0usize;

    while index < dataset.len() {
        let batch_size = (dataset.len() - index).min(BATCH_SIZE);
        let items: Vec<_> = (0..batch_size)
            .map(|offset| dataset.get(index + offset).unwrap())
            .collect();
        let batch = MnistBatch::<TrainingBackend>::from_items(device, &items);
        let logits = model.forward(batch.images.clone());
        let loss = loss_fn.forward(logits.clone(), batch.labels.clone());
        let (correct, total) = accuracy_counts(logits, batch.labels.clone());

        total_loss += loss.into_scalar().elem::<f32>() * total as f32;
        total_correct += correct;
        total_samples += total;
        batches += 1;
        index += batch_size;

        if let Some(max) = max_batches {
            if batches >= max {
                break;
            }
        }
    }

    let average_loss = total_loss / total_samples as f32;
    let accuracy = total_correct as f32 / total_samples as f32 * 100.0;

    EvaluationMetrics {
        loss: average_loss,
        accuracy,
    }
}

fn accuracy_counts<B: Backend>(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> (usize, usize) {
    let predictions = logits.argmax(1).squeeze(1);
    let correct = predictions
        .equal(targets.clone())
        .int()
        .sum()
        .into_scalar()
        .elem::<i64>() as usize;
    let total = targets.dims()[0];

    (correct, total)
}

fn sample_predictions(
    model: &MnistClassifier<TrainingBackend>,
    device: &CandleDevice,
    dataset: &MnistDataset,
    count: usize,
) -> Result<Vec<SamplePrediction>> {
    let available = count.min(dataset.len());
    if available == 0 {
        return Ok(Vec::new());
    }

    let mut indices = Vec::with_capacity(available);
    let mut items = Vec::with_capacity(available);
    for index in 0..available {
        let item = dataset
            .get(index)
            .ok_or_else(|| anyhow!("dataset index {} out of bounds", index))?;
        indices.push(index);
        items.push(item);
    }

    let batch = MnistBatch::<TrainingBackend>::from_items(device, &items);
    let logits = model.forward(batch.images.clone());
    let predictions = logits
        .argmax(1)
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|err| anyhow!("failed to decode predictions: {err:?}"))?;

    let mut samples = Vec::with_capacity(available);
    for (i, prediction) in predictions.iter().enumerate() {
        let item = &items[i];
        let image_data_url = encode_item_image(item)?;
        samples.push(SamplePrediction {
            index: indices[i],
            label: item.label as usize,
            prediction: *prediction as usize,
            image_data_url,
        });
    }

    Ok(samples)
}

fn encode_item_image(item: &burn_dataset::vision::MnistItem) -> Result<String> {
    let mut pixels = Vec::with_capacity(INPUT_DIM);
    for row in item.image.iter() {
        for &pixel in row.iter() {
            pixels.push(pixel as f32 / 255.0);
        }
    }

    encode_luma_png_data_url(28, 28, &pixels)
}

fn linear_from_rng<B: Backend>(
    rng: &mut StdRng,
    device: &B::Device,
    fan_in: usize,
    fan_out: usize,
) -> Linear<B> {
    let limit = (1.0f32 / fan_in as f32).sqrt();
    let weight = random_tensor::<B, 2>(rng, [fan_in, fan_out], limit, device);
    let bias = random_tensor::<B, 1>(rng, [fan_out], limit, device);

    Linear {
        weight: Param::from_tensor(weight),
        bias: Some(Param::from_tensor(bias)),
    }
}

fn random_tensor<B: Backend, const D: usize>(
    rng: &mut StdRng,
    shape: [usize; D],
    limit: f32,
    device: &B::Device,
) -> Tensor<B, D> {
    let total: usize = shape.iter().product();
    let mut values = Vec::with_capacity(total);

    for _ in 0..total {
        let sample = rng.gen::<f32>() * 2.0 * limit - limit;
        values.push(sample);
    }

    Tensor::<B, D>::from_floats(TensorData::new(values, shape), device)
}

fn load_benchmark(path: &Path) -> Result<Option<BenchmarkSnapshot>> {
    if path.exists() {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read benchmark from {}", path.display()))?;
        let snapshot = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse benchmark at {}", path.display()))?;
        Ok(Some(snapshot))
    } else {
        Ok(None)
    }
}

fn save_benchmark(path: &Path, snapshot: &BenchmarkSnapshot) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let serialized = serde_json::to_string_pretty(snapshot)?;
    fs::write(path, serialized)
        .with_context(|| format!("failed to write benchmark to {}", path.display()))?;
    Ok(())
}

fn validate_benchmark(actual: &BenchmarkSnapshot, reference: &BenchmarkSnapshot) -> Result<()> {
    ensure_close(
        actual.final_train.step as f32,
        reference.final_train.step as f32,
        0.5,
        "train steps",
    )?;
    ensure_close(
        actual.final_train.loss,
        reference.final_train.loss,
        BENCHMARK_TOLERANCE,
        "final train loss",
    )?;
    ensure_close(
        actual.final_train.accuracy,
        reference.final_train.accuracy,
        BENCHMARK_TOLERANCE,
        "final train accuracy",
    )?;
    ensure_close(
        actual.final_test.loss,
        reference.final_test.loss,
        BENCHMARK_TOLERANCE,
        "final test loss",
    )?;
    ensure_close(
        actual.final_test.accuracy,
        reference.final_test.accuracy,
        BENCHMARK_TOLERANCE,
        "final test accuracy",
    )?;

    match (&actual.five_shot, &reference.five_shot) {
        (Some(actual), Some(reference)) => {
            ensure_close(
                actual.loss,
                reference.loss,
                BENCHMARK_TOLERANCE,
                "five-shot loss",
            )?;
            ensure_close(
                actual.accuracy,
                reference.accuracy,
                BENCHMARK_TOLERANCE,
                "five-shot accuracy",
            )?;
        }
        (None, None) => {}
        _ => {
            return Err(anyhow!(
                "five-shot availability changed between runs; update benchmark if this is intentional"
            ));
        }
    }

    Ok(())
}

fn ensure_close(actual: f32, expected: f32, tolerance: f32, label: &str) -> Result<()> {
    if (actual - expected).abs() > tolerance {
        Err(anyhow!(
            "{} deviated from benchmark (actual {:.4} vs expected {:.4}, tol {:.4})",
            label,
            actual,
            expected,
            tolerance
        ))
    } else {
        Ok(())
    }
}

impl<B: AutodiffBackend> MnistBatch<B> {
    fn from_items(device: &B::Device, items: &[burn_dataset::vision::MnistItem]) -> Self {
        let mut images = Vec::with_capacity(items.len() * INPUT_DIM);
        let mut labels = Vec::with_capacity(items.len());

        for item in items {
            for row in item.image.iter() {
                for &pixel in row.iter() {
                    images.push(pixel as f32 / 255.0);
                }
            }
            labels.push(item.label as i64);
        }

        let images =
            Tensor::<B, 2>::from_floats(TensorData::new(images, [items.len(), INPUT_DIM]), device);
        let labels = Tensor::<B, 1, Int>::from_ints(TensorData::new(labels, [items.len()]), device);

        Self { images, labels }
    }
}
