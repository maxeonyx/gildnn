from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEVICE = torch.device("cpu")
BASE_SEED = 1729


@dataclass(frozen=True)
class TestResult:
    name: str
    passed: bool
    duration_seconds: float
    metrics: dict[str, float | dict[float, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def tensor_batch(tensor: Tensor, batch_size: int) -> Tensor:
    indices = torch.randint(0, tensor.shape[0], (batch_size,), device=tensor.device)
    return tensor.index_select(0, indices)


def paired_batch(inputs: Tensor, targets: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
    indices = torch.randint(0, inputs.shape[0], (batch_size,), device=inputs.device)
    return inputs.index_select(0, indices), targets.index_select(0, indices)


def mean_cosine_similarity(predictions: Tensor, targets: Tensor) -> float:
    return F.cosine_similarity(predictions, targets, dim=-1).mean().item()


def format_metric(value: float) -> str:
    return f"{value:.6f}"


def print_header(name: str) -> None:
    print(f"=== {name} ===")


def print_verdict(result: TestResult) -> None:
    verdict = "PASS" if result.passed else "FAIL"
    print(f"VERDICT {result.name}: {verdict} ({result.duration_seconds:.2f}s)")
    print()


class FrozenTeacher(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.norm(self.fc2(F.gelu(self.fc1(inputs))))


class StudentPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, repr_dim: int, output_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(repr_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.head(self.trunk(inputs))


class FixedTargetStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.head = nn.Linear(output_dim, output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        representation = self.norm(self.fc2(F.gelu(self.fc1(inputs))))
        return self.head(representation)


def run_fixed_target() -> TestResult:
    set_seed(BASE_SEED)
    start = time.perf_counter()
    print_header("fixed-target")

    input_dim = 64
    hidden_dim = 64
    train_inputs = torch.randn(2048, input_dim, device=DEVICE)
    val_inputs = torch.randn(512, input_dim, device=DEVICE)

    teacher = FrozenTeacher(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(DEVICE)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        train_targets = teacher(train_inputs)
        val_targets = teacher(val_inputs)

    student = FixedTargetStudent(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(DEVICE)
    with torch.no_grad():
        student.fc1.weight.copy_(teacher.fc1.weight + 0.05 * torch.randn_like(teacher.fc1.weight))
        student.fc1.bias.copy_(teacher.fc1.bias + 0.05 * torch.randn_like(teacher.fc1.bias))
        student.fc2.weight.copy_(teacher.fc2.weight + 0.05 * torch.randn_like(teacher.fc2.weight))
        student.fc2.bias.copy_(teacher.fc2.bias + 0.05 * torch.randn_like(teacher.fc2.bias))
        student.norm.weight.copy_(teacher.norm.weight)
        student.norm.bias.copy_(teacher.norm.bias)
        student.head.weight.copy_(torch.eye(input_dim, device=DEVICE) + 0.05 * torch.randn(input_dim, input_dim, device=DEVICE))
        student.head.bias.zero_()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)

    with torch.no_grad():
        initial_train_mse = F.mse_loss(student(train_inputs), train_targets).item()
        initial_val_predictions = student(val_inputs)
        initial_val_mse = F.mse_loss(initial_val_predictions, val_targets).item()

    for _ in range(300):
        batch_inputs, batch_targets = paired_batch(train_inputs, train_targets, batch_size=128)
        predictions = student(batch_inputs)
        loss = F.mse_loss(predictions, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_train_mse = F.mse_loss(student(train_inputs), train_targets).item()
        final_val_predictions = student(val_inputs)
        final_val_mse = F.mse_loss(final_val_predictions, val_targets).item()
        final_cosine = mean_cosine_similarity(final_val_predictions, val_targets)

    drop_ratio = initial_val_mse / final_val_mse
    passed = drop_ratio > 20.0 and final_cosine > 0.98 and final_val_mse < initial_train_mse

    print(
        "start train_mse="
        f"{format_metric(initial_train_mse)} val_mse={format_metric(initial_val_mse)}"
    )
    print(
        "end   train_mse="
        f"{format_metric(final_train_mse)} val_mse={format_metric(final_val_mse)} "
        f"drop_ratio={drop_ratio:.2f} cosine={final_cosine:.4f}"
    )

    result = TestResult(
        name="fixed-target",
        passed=passed,
        duration_seconds=time.perf_counter() - start,
        metrics={
            "initial_train_mse": initial_train_mse,
            "initial_val_mse": initial_val_mse,
            "final_train_mse": final_train_mse,
            "final_val_mse": final_val_mse,
            "drop_ratio": drop_ratio,
            "final_cosine": final_cosine,
        },
    )
    print_verdict(result)
    return result


def make_classification_dataset(
    *,
    num_samples: int,
    num_classes: int,
    dim: int,
    prototypes: Tensor,
    clean_noise_scale: float,
    lateral_noise_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    labels = torch.randint(0, num_classes, (num_samples,), device=DEVICE)
    clean = prototypes.index_select(0, labels) + clean_noise_scale * torch.randn(num_samples, dim, device=DEVICE)
    lateral = lateral_noise_scale * torch.randn(num_samples, dim, device=DEVICE)
    return clean, lateral, labels


def train_classifier_for_alpha(
    *,
    alpha: float,
    train_clean: Tensor,
    train_lateral: Tensor,
    train_labels: Tensor,
    val_clean: Tensor,
    val_lateral: Tensor,
    val_labels: Tensor,
    num_classes: int,
) -> tuple[float, float, float]:
    classifier = nn.Linear(train_clean.shape[1], num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=8e-2)

    with torch.no_grad():
        val_inputs = ((1.0 - alpha) * val_clean) + (alpha * val_lateral)
        initial_accuracy = (classifier(val_inputs).argmax(dim=-1) == val_labels).float().mean().item()

    for _ in range(100):
        batch_clean, batch_labels = paired_batch(train_clean, train_labels, batch_size=256)
        batch_lateral = tensor_batch(train_lateral, batch_size=256)
        mixed = ((1.0 - alpha) * batch_clean) + (alpha * batch_lateral)
        logits = classifier(mixed)
        loss = F.cross_entropy(logits, batch_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_inputs = ((1.0 - alpha) * val_clean) + (alpha * val_lateral)
        logits = classifier(val_inputs)
        final_loss = F.cross_entropy(logits, val_labels).item()
        final_accuracy = (logits.argmax(dim=-1) == val_labels).float().mean().item()

    return initial_accuracy, final_accuracy, final_loss


def run_mixing_damage() -> TestResult:
    set_seed(BASE_SEED + 1)
    start = time.perf_counter()
    print_header("mixing-damage")

    num_classes = 16
    dim = 64
    prototypes = F.normalize(torch.randn(num_classes, dim, device=DEVICE), dim=-1) * 8.0
    train_clean, train_lateral, train_labels = make_classification_dataset(
        num_samples=4096,
        num_classes=num_classes,
        dim=dim,
        prototypes=prototypes,
        clean_noise_scale=0.20,
        lateral_noise_scale=8.0,
    )
    val_clean, val_lateral, val_labels = make_classification_dataset(
        num_samples=1024,
        num_classes=num_classes,
        dim=dim,
        prototypes=prototypes,
        clean_noise_scale=0.20,
        lateral_noise_scale=8.0,
    )

    accuracies: dict[float, float] = {}
    initial_accuracies: dict[float, float] = {}
    losses: dict[float, float] = {}

    for alpha in (0.0, 0.1, 0.25, 0.5):
        initial_accuracy, final_accuracy, final_loss = train_classifier_for_alpha(
            alpha=alpha,
            train_clean=train_clean,
            train_lateral=train_lateral,
            train_labels=train_labels,
            val_clean=val_clean,
            val_lateral=val_lateral,
            val_labels=val_labels,
            num_classes=num_classes,
        )
        initial_accuracies[alpha] = initial_accuracy
        accuracies[alpha] = final_accuracy
        losses[alpha] = final_loss
        print(
            f"alpha={alpha:>4.2f} start_acc={initial_accuracy:.4f} "
            f"end_acc={final_accuracy:.4f} end_loss={final_loss:.4f}"
        )

    baseline_accuracy = accuracies[0.0]
    low_mix_gap = max(abs(baseline_accuracy - accuracies[0.1]), abs(baseline_accuracy - accuracies[0.0]))
    high_mix_drop = baseline_accuracy - accuracies[0.5]
    passed = baseline_accuracy > 0.95 and high_mix_drop > 0.15 and low_mix_gap < 0.05

    result = TestResult(
        name="mixing-damage",
        passed=passed,
        duration_seconds=time.perf_counter() - start,
        metrics={
            "baseline_accuracy": baseline_accuracy,
            "alpha_0_1_accuracy": accuracies[0.1],
            "alpha_0_25_accuracy": accuracies[0.25],
            "alpha_0_5_accuracy": accuracies[0.5],
            "high_mix_drop": high_mix_drop,
            "low_mix_gap": low_mix_gap,
        },
    )
    print_verdict(result)
    return result


def make_recurrence_batch(batch_size: int, seq_len: int, vocab_size: int) -> tuple[Tensor, Tensor]:
    first = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
    second = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
    sequence = [first, second]
    while len(sequence) < seq_len:
        sequence.append((sequence[-1] + sequence[-2]) % vocab_size)
    stacked = torch.stack(sequence, dim=1)
    return stacked[:, :-1], stacked[:, -1]


class TinyTeacher(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, input_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.hidden = nn.Sequential(
            nn.Linear(input_len * hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.embedding(tokens)
        hidden = self.hidden(embeddings.flatten(start_dim=1))
        return hidden, self.output(hidden)


class TinyStudent(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, input_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_len * hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        embeddings = self.embedding(tokens)
        return self.net(embeddings.flatten(start_dim=1))


def clone_teacher(teacher: TinyTeacher, vocab_size: int, hidden_dim: int, input_len: int) -> TinyTeacher:
    cloned = TinyTeacher(vocab_size=vocab_size, hidden_dim=hidden_dim, input_len=input_len).to(DEVICE)
    cloned.load_state_dict(teacher.state_dict())
    return cloned


def train_teacher(teacher: TinyTeacher, *, steps: int, lr: float, batch_size: int, seq_len: int, vocab_size: int) -> float:
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    last_loss = 0.0
    for _ in range(steps):
        tokens, targets = make_recurrence_batch(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)
        _, logits = teacher(tokens)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss


def evaluate_tracking(student: TinyStudent, teacher: TinyTeacher, val_tokens: Tensor) -> float:
    with torch.no_grad():
        teacher_hidden, _ = teacher(val_tokens)
        student_hidden = student(val_tokens)
        return F.mse_loss(student_hidden, teacher_hidden).item()


def train_tracking_condition(
    *,
    base_teacher: TinyTeacher,
    teacher_moves: bool,
    moving_teacher_lr: float,
    tracking_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int,
) -> tuple[float, float, float, float]:
    teacher = clone_teacher(base_teacher, vocab_size=vocab_size, hidden_dim=hidden_dim, input_len=seq_len - 1)
    student = TinyStudent(vocab_size=vocab_size, hidden_dim=hidden_dim, input_len=seq_len - 1).to(DEVICE)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=4e-3)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=moving_teacher_lr) if teacher_moves else None

    val_tokens, val_targets = make_recurrence_batch(batch_size=1024, seq_len=seq_len, vocab_size=vocab_size)
    initial_mse = evaluate_tracking(student, teacher, val_tokens)

    teacher_loss_before = 0.0
    teacher_loss_after = 0.0
    if teacher_moves:
        with torch.no_grad():
            _, logits = teacher(val_tokens)
            teacher_loss_before = F.cross_entropy(logits, val_targets).item()

    for _ in range(tracking_steps):
        tokens, targets = make_recurrence_batch(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)
        if teacher_moves and teacher_optimizer is not None:
            teacher_hidden, teacher_logits = teacher(tokens)
            teacher_loss = F.cross_entropy(teacher_logits, targets)
            teacher_optimizer.zero_grad(set_to_none=True)
            teacher_loss.backward()
            teacher_optimizer.step()
            with torch.no_grad():
                teacher_hidden = teacher(tokens)[0]
        else:
            with torch.no_grad():
                teacher_hidden = teacher(tokens)[0]
        student_hidden = student(tokens)
        tracking_loss = F.mse_loss(student_hidden, teacher_hidden)
        student_optimizer.zero_grad(set_to_none=True)
        tracking_loss.backward()
        student_optimizer.step()

    final_mse = evaluate_tracking(student, teacher, val_tokens)
    if teacher_moves:
        with torch.no_grad():
            _, logits = teacher(val_tokens)
            teacher_loss_after = F.cross_entropy(logits, val_targets).item()

    return initial_mse, final_mse, teacher_loss_before, teacher_loss_after


def run_moving_target() -> TestResult:
    set_seed(BASE_SEED + 2)
    start = time.perf_counter()
    print_header("moving-target")

    vocab_size = 17
    hidden_dim = 32
    seq_len = 6

    teacher = TinyTeacher(vocab_size=vocab_size, hidden_dim=hidden_dim, input_len=seq_len - 1).to(DEVICE)
    pretrain_loss = train_teacher(
        teacher,
        steps=100,
        lr=1e-2,
        batch_size=256,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

    frozen_initial, frozen_final, _, _ = train_tracking_condition(
        base_teacher=teacher,
        teacher_moves=False,
        moving_teacher_lr=0.0,
        tracking_steps=200,
        batch_size=256,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )
    moving_initial, moving_final, teacher_loss_before, teacher_loss_after = train_tracking_condition(
        base_teacher=teacher,
        teacher_moves=True,
        moving_teacher_lr=5e-4,
        tracking_steps=200,
        batch_size=256,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
    )

    ratio = moving_final / frozen_final
    passed = frozen_final < frozen_initial and moving_final < moving_initial and ratio <= 2.0

    print(f"teacher pretrain final_ce={pretrain_loss:.4f}")
    print(
        f"frozen start_mse={frozen_initial:.6f} end_mse={frozen_final:.6f} "
        f"improvement={frozen_initial / frozen_final:.2f}x"
    )
    print(
        f"moving start_mse={moving_initial:.6f} end_mse={moving_final:.6f} ratio_vs_frozen={ratio:.2f} "
        f"teacher_ce={teacher_loss_before:.4f}->{teacher_loss_after:.4f}"
    )

    result = TestResult(
        name="moving-target",
        passed=passed,
        duration_seconds=time.perf_counter() - start,
        metrics={
            "teacher_pretrain_ce": pretrain_loss,
            "frozen_initial_mse": frozen_initial,
            "frozen_final_mse": frozen_final,
            "moving_initial_mse": moving_initial,
            "moving_final_mse": moving_final,
            "moving_vs_frozen_ratio": ratio,
        },
    )
    print_verdict(result)
    return result


def make_topology_dataset(num_samples: int, dim: int, block0_encoder: FrozenTeacher) -> tuple[Tensor, Tensor, Tensor]:
    latent = torch.randn(num_samples, dim, device=DEVICE)
    block0_input = latent + (0.15 * torch.randn(num_samples, dim, device=DEVICE))
    block1_noise = torch.randn(num_samples, dim, device=DEVICE)
    with torch.no_grad():
        block0_representation = block0_encoder(block0_input)
    return block0_representation, block0_input, block1_noise


def build_topology_input(wiring: str, block0_representation: Tensor, block1_noise: Tensor) -> Tensor:
    if wiring == "0->1":
        return torch.cat([block1_noise, block0_representation], dim=-1)
    if wiring in {"1->0", "none"}:
        return torch.cat([block1_noise, torch.zeros_like(block1_noise)], dim=-1)
    raise ValueError(f"Unknown wiring {wiring!r}")


def train_topology_case(
    *,
    wiring: str,
    train_targets: Tensor,
    train_noise: Tensor,
    val_targets: Tensor,
    val_noise: Tensor,
) -> tuple[float, float]:
    predictor = StudentPredictor(
        input_dim=train_noise.shape[1] * 2,
        hidden_dim=64,
        repr_dim=32,
        output_dim=train_targets.shape[1],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=4e-3)

    with torch.no_grad():
        initial_inputs = build_topology_input(wiring, val_targets, val_noise)
        initial_mse = F.mse_loss(predictor(initial_inputs), val_targets).item()

    for _ in range(200):
        batch_noise, batch_targets = paired_batch(train_noise, train_targets, batch_size=256)
        predictor_inputs = build_topology_input(wiring, batch_targets, batch_noise)
        predictions = predictor(predictor_inputs)
        loss = F.mse_loss(predictions, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_inputs = build_topology_input(wiring, val_targets, val_noise)
        final_mse = F.mse_loss(predictor(final_inputs), val_targets).item()

    return initial_mse, final_mse


def run_topology() -> TestResult:
    set_seed(BASE_SEED + 3)
    start = time.perf_counter()
    print_header("topology")

    dim = 32
    block0_encoder = FrozenTeacher(input_dim=dim, hidden_dim=64, output_dim=dim).to(DEVICE)
    block0_encoder.eval()
    for parameter in block0_encoder.parameters():
        parameter.requires_grad_(False)

    train_targets, _, train_noise = make_topology_dataset(4096, dim, block0_encoder)
    val_targets, _, val_noise = make_topology_dataset(1024, dim, block0_encoder)

    final_mses: dict[str, float] = {}
    initial_mses: dict[str, float] = {}

    for wiring in ("0->1", "1->0", "none"):
        initial_mse, final_mse = train_topology_case(
            wiring=wiring,
            train_targets=train_targets,
            train_noise=train_noise,
            val_targets=val_targets,
            val_noise=val_noise,
        )
        initial_mses[wiring] = initial_mse
        final_mses[wiring] = final_mse
        print(
            f"wiring={wiring:>4} start_mse={initial_mse:.6f} end_mse={final_mse:.6f} "
            f"improvement={initial_mse / final_mse:.2f}x"
        )

    best_wrong = min(final_mses["1->0"], final_mses["none"])
    separation = best_wrong / final_mses["0->1"]
    passed = separation > 5.0

    result = TestResult(
        name="topology",
        passed=passed,
        duration_seconds=time.perf_counter() - start,
        metrics={
            "forward_final_mse": final_mses["0->1"],
            "reverse_final_mse": final_mses["1->0"],
            "none_final_mse": final_mses["none"],
            "separation": separation,
        },
    )
    print_verdict(result)
    return result


TESTS = {
    "fixed-target": run_fixed_target,
    "mixing-damage": run_mixing_damage,
    "moving-target": run_moving_target,
    "topology": run_topology,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_name", choices=[*TESTS.keys(), "all"])
    return parser.parse_args()


def main() -> int:
    torch.set_num_threads(1)
    args = parse_args()
    selected = list(TESTS.items()) if args.test_name == "all" else [(args.test_name, TESTS[args.test_name])]
    results = [runner() for _, runner in selected]

    all_passed = all(result.passed for result in results)
    if len(results) > 1:
        print("=== summary ===")
        for result in results:
            verdict = "PASS" if result.passed else "FAIL"
            print(f"{result.name}: {verdict} ({result.duration_seconds:.2f}s)")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
