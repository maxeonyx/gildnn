use std::{
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
use burn_dataset::{
    vision::{MnistDataset, MnistItem},
    Dataset,
};
use gildnn_core::{
    ensure_report_file, load_or_init, png_data_uri, seeded_rng, update_sections, EvaluationMetrics,
    ExperimentMode, ExperimentModeArgs, ReportSection, StepMetrics,
};
use mnist_core::{
    dataset::{collect_first_items, item_to_normalized_pixels, sample_random_items},
    image::{
        build_dataset_row_image, build_final_evaluation_panels, build_hypothesis_panel,
        build_prediction_panel, EvaluationPanel,
    },
    schedule::prediction_schedule,
    DEFAULT_PANEL_DIGITS, MNIST_PIXEL_COUNT,
};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};

const INPUT_DIM: usize = MNIST_PIXEL_COUNT;
const NUM_CLASSES: usize = 10;
const HIDDEN_DIM: usize = 128;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 1e-3;
const TEST_FIVE_SHOT_BATCHES: usize = 5;
const FIVE_SHOT_STEP: usize = 10;
const FULL_TRAIN_STEPS: usize = 200;
const TEST_TRAIN_STEPS: usize = 25;
const BENCHMARK_TOLERANCE: f32 = 5e-3;
const DATASET_ROW_COUNT: usize = DEFAULT_PANEL_DIGITS;
const HYPOTHESIS_SAMPLE_COUNT: usize = 50;
const SNAPSHOT_SAMPLE_COUNT: usize = 10;
const MAX_SNAPSHOT_IMAGES: usize = 10;

type TrainingBackend = Autodiff<Candle<f32, i64>>;

#[derive(Serialize, Deserialize)]
struct ExperimentConfig {
    seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchmarkSnapshot {
    #[serde(default = "default_benchmark_train_steps")]
    train_steps: usize,
    final_train: StepMetrics,
    final_test: EvaluationMetrics,
    five_shot: Option<EvaluationMetrics>,
}

const fn default_benchmark_train_steps() -> usize {
    TEST_TRAIN_STEPS
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
    dataset_train_row: Option<Vec<u8>>,
    dataset_test_row: Option<Vec<u8>>,
    hypothesis_panels: Vec<Vec<u8>>,
    training_snapshots: Vec<TrainingSnapshotArtifact>,
    final_evaluation: Vec<EvaluationPanelArtifact>,
}

struct TrainingSnapshotArtifact {
    step: usize,
    correct: usize,
    total: usize,
    image_bytes: Vec<u8>,
}

struct EvaluationPanelArtifact {
    index: usize,
    correct: usize,
    total: usize,
    image_bytes: Vec<u8>,
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
    let mode_args = ExperimentModeArgs::parse_from_env()?;
    if mode_args.help_requested() {
        print_usage();
        return Ok(());
    }
    let mode = mode_args.mode();

    let paths = initialize_paths()?;
    let config: ExperimentConfig = load_or_init(&paths.config, || ExperimentConfig { seed: 1337 })?;
    ensure_report_file(&paths.report)?;
    let benchmark = load_benchmark(&paths.benchmark)?;

    let train_steps = mode.select(FULL_TRAIN_STEPS, TEST_TRAIN_STEPS);
    println!("running MNIST baseline in {} mode", mode.label());

    let result = run_training(train_steps, &config)?;

    if matches!(mode, ExperimentMode::Full) {
        write_report(&paths.report, &config, train_steps, &result)?;
    }

    match mode {
        ExperimentMode::Full => {
            if benchmark.is_none() {
                println!(
                    "no benchmark snapshot recorded yet; run with --mode test to capture one."
                );
            }
        }
        ExperimentMode::Test => {
            let snapshot = BenchmarkSnapshot {
                train_steps,
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
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
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

    let dataset_train_items = collect_first_items(&train_dataset, DATASET_ROW_COUNT)?;
    let dataset_test_items = collect_first_items(&test_dataset, DATASET_ROW_COUNT)?;
    let dataset_train_row = if dataset_train_items.is_empty() {
        None
    } else {
        Some(build_dataset_row_image(&dataset_train_items)?)
    };
    let dataset_test_row = if dataset_test_items.is_empty() {
        None
    } else {
        Some(build_dataset_row_image(&dataset_test_items)?)
    };

    let mut artifact_rng = seeded_rng(config.seed ^ 0x9e37_79b9_7f4a_7c15);
    let hypothesis_items =
        sample_random_items(&test_dataset, HYPOTHESIS_SAMPLE_COUNT, &mut artifact_rng)?;
    let hypothesis_panels = hypothesis_items
        .chunks(DATASET_ROW_COUNT)
        .filter(|chunk| !chunk.is_empty())
        .map(build_hypothesis_panel)
        .collect::<Result<Vec<_>>>()?;
    let hypothesis_inputs: Vec<Vec<f32>> = hypothesis_items
        .iter()
        .map(item_to_normalized_pixels)
        .collect();
    let hypothesis_labels: Vec<usize> = hypothesis_items
        .iter()
        .map(|item| item.label as usize)
        .collect();

    let snapshot_items =
        sample_random_items(&test_dataset, SNAPSHOT_SAMPLE_COUNT, &mut artifact_rng)?;
    let snapshot_inputs: Vec<Vec<f32>> = snapshot_items
        .iter()
        .map(item_to_normalized_pixels)
        .collect();
    let snapshot_labels: Vec<usize> = snapshot_items
        .iter()
        .map(|item| item.label as usize)
        .collect();
    let schedule_limit = MAX_SNAPSHOT_IMAGES.saturating_sub(1);
    let snapshot_schedule = if schedule_limit == 0 {
        Vec::new()
    } else {
        prediction_schedule(train_steps, schedule_limit)
    };
    let mut schedule_iter = snapshot_schedule.iter().copied();
    let mut next_snapshot_step = schedule_iter.next();
    let mut training_snapshots = Vec::with_capacity(snapshot_schedule.len() + 1);

    if !snapshot_labels.is_empty() {
        let predictions = infer_predictions(&model, &device, &snapshot_items)?;
        let panel = build_prediction_panel(&snapshot_inputs, &snapshot_labels, &predictions)?;
        let correct_predictions = predictions
            .iter()
            .zip(&snapshot_labels)
            .filter(|(pred, label)| pred == label)
            .count();
        training_snapshots.push(TrainingSnapshotArtifact {
            step: 0,
            correct: correct_predictions,
            total: snapshot_labels.len(),
            image_bytes: panel,
        });
    }

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

        if Some(step + 1) == next_snapshot_step {
            let predictions = infer_predictions(&model, &device, &snapshot_items)?;
            let panel = build_prediction_panel(&snapshot_inputs, &snapshot_labels, &predictions)?;
            let correct_predictions = predictions
                .iter()
                .zip(&snapshot_labels)
                .filter(|(pred, label)| pred == label)
                .count();
            training_snapshots.push(TrainingSnapshotArtifact {
                step: step + 1,
                correct: correct_predictions,
                total: snapshot_labels.len(),
                image_bytes: panel,
            });
            next_snapshot_step = schedule_iter.next();
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

    let final_predictions = infer_predictions(&model, &device, &hypothesis_items)?;
    let evaluation_panels = build_final_evaluation_panels(
        &hypothesis_inputs,
        &hypothesis_labels,
        &final_predictions,
        DATASET_ROW_COUNT,
    )?;
    let final_evaluation = evaluation_panels
        .into_iter()
        .map(|panel: EvaluationPanel| EvaluationPanelArtifact {
            index: panel.index,
            correct: panel.correct,
            total: panel.total,
            image_bytes: panel.image_bytes,
        })
        .collect();

    Ok(ExperimentResult {
        history,
        five_shot,
        final_test: final_eval,
        final_train,
        dataset_train_row,
        dataset_test_row,
        hypothesis_panels,
        training_snapshots,
        final_evaluation,
    })
}

fn write_report(
    report_path: &Path,
    config: &ExperimentConfig,
    train_steps: usize,
    result: &ExperimentResult,
) -> Result<()> {
    cleanup_legacy_assets(report_path)?;

    let dataset_train_section = render_dataset_section(
        "Training split",
        &result.dataset_train_row,
        &["dataset", "train", "row"],
    )?;
    let dataset_test_section = render_dataset_section(
        "Test split",
        &result.dataset_test_row,
        &["dataset", "test", "row"],
    )?;
    let hypothesis_section = render_hypothesis_section(&result.hypothesis_panels)?;
    let training_section =
        render_training_progress_section(&result.training_snapshots, &result.final_train)?;
    let final_evaluation_section = render_final_evaluation_section(&result.final_evaluation)?;

    let sections = [
        ReportSection::new("dataset-train-row", dataset_train_section),
        ReportSection::new("dataset-test-row", dataset_test_section),
        ReportSection::new("hypothesis", hypothesis_section),
        ReportSection::new(
            "configuration",
            render_configuration_section(config, train_steps),
        ),
        ReportSection::new(
            "metrics",
            render_metrics_section(&result.history, result.five_shot, &result.final_test),
        ),
        ReportSection::new("training-progress", training_section),
        ReportSection::new("final-evaluation", final_evaluation_section),
    ];

    update_sections(report_path, &sections)
}

fn cleanup_legacy_assets(report_path: &Path) -> Result<()> {
    if let Some(dir) = report_path
        .parent()
        .map(|parent| parent.join("report_assets"))
    {
        if dir.exists() {
            fs::remove_dir_all(&dir).with_context(|| {
                format!("failed to remove legacy assets directory {}", dir.display())
            })?;
        }
    }
    Ok(())
}

fn render_dataset_section(
    label: &str,
    image: &Option<Vec<u8>>,
    slug_parts: &[&str],
) -> Result<String> {
    match image {
        Some(bytes) => {
            let reference = render_image_reference(slug_parts, bytes, &format!("{label} digits"))?;
            Ok(format!(
                "Each strip shows ten MNIST digits sampled from the {} in dataset order.\n\n{reference}\n",
                label.to_lowercase(),
            ))
        }
        None => {
            let lower = label.to_lowercase();
            Ok(format!("No {lower} examples available.\n"))
        }
    }
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

fn render_hypothesis_section(panels: &[Vec<u8>]) -> Result<String> {
    if panels.is_empty() {
        return Ok("No hypothesis visualization available.\n".to_string());
    }

    let mut output = String::new();
    let _ = writeln!(
        &mut output,
        "#### Hypothesis images ({} batches)\n",
        panels.len()
    );
    output.push_str(
        "Each hypothesis panel stacks the MNIST input row (top) with the expected digit rendered as a label row (bottom).\n\n",
    );

    for (index, panel) in panels.iter().enumerate() {
        let reference = render_image_reference(
            &["hypothesis", &format!("batch-{}", index + 1)],
            panel,
            &format!("Hypothesis panel {}", index + 1),
        )?;
        let _ = writeln!(&mut output, "{}", reference);
    }
    output.push('\n');
    Ok(output)
}

fn render_training_progress_section(
    snapshots: &[TrainingSnapshotArtifact],
    final_train: &StepMetrics,
) -> Result<String> {
    if snapshots.is_empty() {
        return Ok("No training snapshots captured.\n".to_string());
    }

    let mut output = String::new();
    let _ = writeln!(&mut output, "#### Training snapshots\n");
    output.push_str(
        "Each snapshot shows inputs (top), expected labels (middle), and model predictions (bottom). Dark green panels mark correct predictions; dark red panels mark errors.\n\n",
    );

    const HIGHLIGHT_STEPS: [usize; 4] = [0, 1, 2, 4];

    for snapshot in snapshots {
        let reference = render_image_reference(
            &["snapshot", &format!("step-{:04}", snapshot.step)],
            &snapshot.image_bytes,
            &format!(
                "Predictions at step {} ({} / {} correct)",
                snapshot.step, snapshot.correct, snapshot.total
            ),
        )?;
        if HIGHLIGHT_STEPS.contains(&snapshot.step) {
            let _ = writeln!(
                &mut output,
                "#### Step {} ({} / {} correct)\n",
                snapshot.step, snapshot.correct, snapshot.total
            );
        }
        let _ = writeln!(&mut output, "{}", reference);
        output.push('\n');
    }

    let _ = writeln!(
        &mut output,
        "Final step {} train accuracy: {:.2}%\n",
        final_train.step, final_train.accuracy
    );

    Ok(output)
}

fn render_final_evaluation_section(panels: &[EvaluationPanelArtifact]) -> Result<String> {
    if panels.is_empty() {
        return Ok("No held-out evaluation visualizations generated.\n".to_string());
    }

    let mut output = String::new();
    let total_correct: usize = panels.iter().map(|panel| panel.correct).sum();
    let total_samples: usize = panels.iter().map(|panel| panel.total).sum();
    let _ = writeln!(
        &mut output,
        "#### Final evaluation ({} panels)\n",
        panels.len()
    );
    let _ = writeln!(
        &mut output,
        "Overall: {} / {} correct across the hypothesis set.\n",
        total_correct, total_samples
    );
    output.push_str(
        "Each evaluation panel mirrors the snapshot layout with dark green marking correct predictions and dark red highlighting errors.\n\n",
    );

    for panel in panels {
        let reference = render_image_reference(
            &["final", &format!("panel-{}", panel.index)],
            &panel.image_bytes,
            &format!(
                "Final evaluation panel {} ({} / {} correct)",
                panel.index, panel.correct, panel.total
            ),
        )?;
        let _ = writeln!(&mut output, "{}", reference);
        output.push('\n');
    }
    output.push('\n');

    Ok(output)
}

fn render_image_reference(parts: &[&str], bytes: &[u8], alt_text: &str) -> Result<String> {
    let slug = slugify_identifier(parts);
    if slug.is_empty() {
        return Err(anyhow!("image reference slug cannot be empty"));
    }
    let uri = png_data_uri(bytes);
    Ok(format!(
        "[{slug}]: <{uri}>\n\n![{alt_text}][{slug}]\n\n",
        slug = slug,
        uri = uri,
        alt_text = alt_text
    ))
}

fn slugify_identifier(parts: &[&str]) -> String {
    let mut segments = Vec::new();
    for part in parts {
        let mut segment = String::new();
        let mut last_dash = true;
        for ch in part.chars() {
            if ch.is_ascii_alphanumeric() {
                segment.push(ch.to_ascii_lowercase());
                last_dash = false;
            } else if !last_dash {
                segment.push('-');
                last_dash = true;
            }
        }
        let segment = segment.trim_matches('-');
        if !segment.is_empty() {
            segments.push(segment.to_string());
        }
    }
    segments.join("-")
}

fn summarize_history<'a>(history: &'a [StepMetrics]) -> Vec<&'a StepMetrics> {
    if history.is_empty() {
        return Vec::new();
    }

    let total_steps = history.last().unwrap().step;
    let mut checkpoints = vec![1, 2, 4, 8, 16, 32, 64, 128, total_steps];
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

fn infer_predictions(
    model: &MnistClassifier<TrainingBackend>,
    device: &CandleDevice,
    items: &[MnistItem],
) -> Result<Vec<usize>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let batch = MnistBatch::<TrainingBackend>::from_items(device, items);
    let logits = model.forward(batch.images.clone());
    let predictions = logits
        .argmax(1)
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|err| anyhow!("failed to decode predictions: {err:?}"))?;

    Ok(predictions
        .into_iter()
        .map(|value| value as usize)
        .collect())
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
    if actual.train_steps != reference.train_steps {
        return Err(anyhow!(
            "train step count mismatch (expected {}, found {})",
            reference.train_steps,
            actual.train_steps
        ));
    }
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
    fn from_items(device: &B::Device, items: &[MnistItem]) -> Self {
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
