from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DEFAULT_SEED = 42
SHORT_CONTEXT = 4
MID_CONTEXT = 32
LONG_CONTEXT = 128
TRAIN_CHARACTERS = 80_000
VAL_CHARACTERS = 20_000
TRAINING_STEPS = 4_800
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 8_192
LEARNING_RATE = 3e-3
D_MODEL = 64
N_HEADS = 2
FF_DIM = 128
N_LAYERS = 1
TEMPERATURE = 1.0
PRINT_INTERVAL = 400
ALL_CONDITIONS = ("block0_alone", "two_blocks")


@dataclass(frozen=True)
class WindowDataset:
    short_inputs: Tensor
    mid_inputs: Tensor
    long_inputs: Tensor
    targets: Tensor
    vocab_size: int


@dataclass(frozen=True)
class EvalResult:
    val_loss: float
    val_accuracy: float


@dataclass(frozen=True)
class ConditionResult:
    name: str
    val_loss: float
    val_accuracy: float
    delta_from_baseline: float
    wall_seconds: float


@dataclass(frozen=True)
class ModelSize:
    d_model: int
    n_heads: int
    ff_dim: int
    n_layers: int


@dataclass(frozen=True)
class Corpus:
    train_text: str
    val_text: str
    stoi: dict[str, int]
    itos: tuple[str, ...]


@dataclass(frozen=True)
class TrainedCondition:
    result: ConditionResult
    model: TiedReadoutModel


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--conditions", type=str, default=",".join(ALL_CONDITIONS))
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--lateral-scale", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--local-loss", choices=("ce", "cosine", "l2"), default="ce")
    parser.add_argument("--d-model", type=positive_int, default=D_MODEL)
    parser.add_argument("--n-heads", type=positive_int, default=N_HEADS)
    parser.add_argument("--ff-dim", type=positive_int, default=FF_DIM)
    parser.add_argument("--n-layers", type=positive_int, default=N_LAYERS)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    return parser.parse_args()


def resolve_model_size(args: argparse.Namespace) -> ModelSize:
    return ModelSize(
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        n_layers=args.n_layers,
    )


def parse_condition_names(raw_conditions: str) -> tuple[str, ...]:
    requested_conditions = tuple(condition.strip() for condition in raw_conditions.split(",") if condition.strip())
    if len(requested_conditions) == 0:
        raise ValueError("conditions must contain at least one condition name")

    invalid_conditions = [condition for condition in requested_conditions if condition not in ALL_CONDITIONS]
    if invalid_conditions:
        raise ValueError(
            "Unknown conditions requested: "
            + ", ".join(invalid_conditions)
            + ". Valid conditions are: "
            + ", ".join(ALL_CONDITIONS)
        )
    return requested_conditions


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset_to_device(dataset: WindowDataset, device: torch.device) -> WindowDataset:
    return WindowDataset(
        short_inputs=dataset.short_inputs.to(device),
        mid_inputs=dataset.mid_inputs.to(device),
        long_inputs=dataset.long_inputs.to(device),
        targets=dataset.targets.to(device),
        vocab_size=dataset.vocab_size,
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def resolve_text_file() -> Path:
    return Path(__file__).resolve().parent.parent / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"


def load_corpus_text() -> str:
    return resolve_text_file().read_text(encoding="utf-8")


def choose_validation_text(text: str, *, train_text: str, val_characters: int) -> str:
    start = len(train_text)
    stop = start + val_characters
    candidate = text[start:stop]
    if len(candidate) < val_characters:
        raise ValueError(f"Need validation slice of {val_characters} characters, got {len(candidate)}.")
    if set(candidate).issubset(set(train_text)):
        return candidate

    max_start = len(text) - val_characters
    for candidate_start in range(start + 1, max_start + 1):
        candidate_stop = candidate_start + val_characters
        candidate = text[candidate_start:candidate_stop]
        if set(candidate).issubset(set(train_text)):
            return candidate

    missing = sorted(set(text[start : max_start + val_characters]) - set(train_text))
    raise ValueError(
        "Could not find a validation slice whose characters are all present in the training slice. "
        f"Missing training vocabulary coverage for: {missing}"
    )


def encode_window_dataset(text: str, *, stoi: dict[str, int] | None = None) -> tuple[Tensor, Tensor, int]:
    required_characters = LONG_CONTEXT + 1
    if len(text) < required_characters:
        raise ValueError(f"Text split must be at least {required_characters} characters long, got {len(text)}.")

    if stoi is None:
        vocab = sorted(set(text))
        stoi = {char: index for index, char in enumerate(vocab)}

    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    inputs = []
    targets = []
    for start in range(len(encoded) - LONG_CONTEXT):
        stop = start + LONG_CONTEXT
        inputs.append(encoded[start:stop])
        targets.append(encoded[stop])

    return torch.stack(inputs), torch.tensor(targets, dtype=torch.long), len(stoi)


def load_window_dataset() -> tuple[WindowDataset, WindowDataset, Corpus]:
    raw_text = load_corpus_text()
    required_characters = TRAIN_CHARACTERS + VAL_CHARACTERS
    if len(raw_text) < required_characters:
        raise ValueError(f"Need at least {required_characters} characters, got {len(raw_text)}.")

    train_text = raw_text[:TRAIN_CHARACTERS]
    val_text = choose_validation_text(raw_text, train_text=train_text, val_characters=VAL_CHARACTERS)
    train_long_inputs, train_targets, vocab_size = encode_window_dataset(train_text)
    train_vocab = sorted(set(train_text))
    stoi = {char: index for index, char in enumerate(train_vocab)}
    itos = tuple(train_vocab)
    missing_val_chars = sorted(set(val_text) - set(train_text))
    if missing_val_chars:
        raise ValueError(f"Validation text contains characters absent from training text: {missing_val_chars}")
    val_long_inputs, val_targets, _ = encode_window_dataset(val_text, stoi=stoi)

    return (
        WindowDataset(
            short_inputs=train_long_inputs[:, -SHORT_CONTEXT:],
            mid_inputs=train_long_inputs[:, -MID_CONTEXT:],
            long_inputs=train_long_inputs,
            targets=train_targets,
            vocab_size=vocab_size,
        ),
        WindowDataset(
            short_inputs=val_long_inputs[:, -SHORT_CONTEXT:],
            mid_inputs=val_long_inputs[:, -MID_CONTEXT:],
            long_inputs=val_long_inputs,
            targets=val_targets,
            vocab_size=vocab_size,
        ),
        Corpus(train_text=train_text, val_text=val_text, stoi=stoi, itos=itos),
    )


def sample_batch(dataset: WindowDataset, *, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = torch.randint(0, dataset.targets.shape[0], (batch_size,), device=device)
    return (
        dataset.short_inputs[indices],
        dataset.mid_inputs[indices],
        dataset.long_inputs[indices],
        dataset.targets[indices],
    )


def next_char_metrics(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, accuracy


def normalize_hidden(hidden: Tensor) -> Tensor:
    return F.normalize(hidden.float(), dim=-1).to(hidden.dtype)


def tied_logits(hidden: Tensor, embedding: nn.Embedding, *, temperature: float, normalize: bool) -> Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if normalize:
        hidden = normalize_hidden(hidden)
        embedding_weight = F.normalize(embedding.weight.float(), dim=-1).to(embedding.weight.dtype)
        return F.linear(hidden, embedding_weight) / temperature
    return F.linear(hidden, embedding.weight) / temperature


def normalized_target_embeddings(embedding: nn.Embedding, targets: Tensor) -> Tensor:
    return F.normalize(embedding.weight[targets].float(), dim=-1).to(embedding.weight.dtype)


def interior_local_loss(
    hidden: Tensor,
    *,
    embedding: nn.Embedding,
    targets: Tensor,
    temperature: float,
    normalize: bool,
    local_loss: str,
) -> Tensor:
    if local_loss == "ce":
        logits = tied_logits(hidden, embedding, temperature=temperature, normalize=normalize)
        return F.cross_entropy(logits, targets)

    target_embeddings = normalized_target_embeddings(embedding, targets)
    if local_loss == "cosine":
        return 1.0 - F.cosine_similarity(hidden.float(), target_embeddings.float(), dim=-1).mean()
    if local_loss == "l2":
        return F.mse_loss(hidden.float(), target_embeddings.float())
    raise ValueError(f"Unsupported local_loss: {local_loss}")


class CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, context_size, d_model = x.shape
        queries, keys, values = self.qkv(x).chunk(3, dim=-1)

        def reshape_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, context_size, self.n_heads, self.head_dim).transpose(1, 2)

        queries = reshape_heads(queries)
        keys = reshape_heads(keys)
        values = reshape_heads(values)
        attended = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, context_size, d_model)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, ff_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, ff_dim)
        self.out_proj = nn.Linear(ff_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(F.gelu(self.in_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ff_dim=ff_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, *, context_size: int, d_model: int, n_heads: int, ff_dim: int, n_layers: int) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        self.context_size = context_size
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim) for _ in range(n_layers)
        )

    def forward(self, inputs: Tensor, token_embedding: nn.Embedding) -> Tensor:
        if inputs.shape[1] != self.context_size:
            raise ValueError(f"Expected context {self.context_size}, got {inputs.shape[1]}")
        positions = torch.arange(self.context_size, device=inputs.device)
        hidden = token_embedding(inputs) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class TiedReadoutModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        model_size: ModelSize,
        use_mid: bool,
        use_long: bool,
        temperature: float,
        lateral_scale: float,
        normalize: bool,
        local_loss: str,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.temperature = temperature
        self.lateral_scale = lateral_scale
        self.normalize = normalize
        self.local_loss = local_loss
        self.token_embedding = nn.Embedding(vocab_size, model_size.d_model)
        self.output_block = SequenceEncoder(
            context_size=SHORT_CONTEXT,
            d_model=model_size.d_model,
            n_heads=model_size.n_heads,
            ff_dim=model_size.ff_dim,
            n_layers=model_size.n_layers,
        )
        self.mid_block = (
            SequenceEncoder(
                context_size=MID_CONTEXT,
                d_model=model_size.d_model,
                n_heads=model_size.n_heads,
                ff_dim=model_size.ff_dim,
                n_layers=model_size.n_layers,
            )
            if use_mid
            else None
        )
        self.long_block = (
            SequenceEncoder(
                context_size=LONG_CONTEXT,
                d_model=model_size.d_model,
                n_heads=model_size.n_heads,
                ff_dim=model_size.ff_dim,
                n_layers=model_size.n_layers,
            )
            if use_long
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.model_size.d_model**-0.5)

    def block_last_hidden(self, block: SequenceEncoder, inputs: Tensor) -> Tensor:
        hidden = block(inputs, self.token_embedding)[:, -1, :]
        if self.normalize:
            return normalize_hidden(hidden)
        return hidden

    def block_last_hidden_only(self, block: SequenceEncoder, inputs: Tensor) -> Tensor:
        return self.block_last_hidden(block, inputs)

    def output_logits(self, short_inputs: Tensor, *, mid_hidden: Tensor | None = None, long_hidden: Tensor | None = None) -> Tensor:
        output_hidden = self.block_last_hidden(self.output_block, short_inputs)
        if mid_hidden is not None:
            output_hidden = output_hidden + self.lateral_scale * mid_hidden.detach()
        if long_hidden is not None:
            output_hidden = output_hidden + self.lateral_scale * long_hidden.detach()
        return tied_logits(
            output_hidden,
            self.token_embedding,
            temperature=self.temperature,
            normalize=self.normalize,
        )


def build_model(
    condition_name: str,
    *,
    vocab_size: int,
    model_size: ModelSize,
    temperature: float,
    lateral_scale: float,
    normalize: bool,
    local_loss: str,
) -> TiedReadoutModel:
    match condition_name:
        case "block0_alone":
            return TiedReadoutModel(
                vocab_size=vocab_size,
                model_size=model_size,
                use_mid=False,
                use_long=False,
                temperature=temperature,
                lateral_scale=lateral_scale,
                normalize=normalize,
                local_loss=local_loss,
            )
        case "two_blocks":
            return TiedReadoutModel(
                vocab_size=vocab_size,
                model_size=model_size,
                use_mid=True,
                use_long=True,
                temperature=temperature,
                lateral_scale=lateral_scale,
                normalize=normalize,
                local_loss=local_loss,
            )
        case _:
            raise ValueError(f"Unsupported condition: {condition_name}")


def make_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=device.type == "cuda")


def sample_from_logits(logits: Tensor, *, sample_temperature: float) -> Tensor:
    if sample_temperature <= 0.0:
        raise ValueError(f"sample_temperature must be positive, got {sample_temperature}")
    probabilities = F.softmax(logits.float() / sample_temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def encode_prompt(prompt: str, *, stoi: dict[str, int]) -> list[int]:
    missing_characters = sorted(set(prompt) - set(stoi))
    if missing_characters:
        raise ValueError(f"Prompt contains characters absent from training vocabulary: {missing_characters}")
    return [stoi[char] for char in prompt]


def prompt_context_size(condition_name: str) -> int:
    match condition_name:
        case "block0_alone":
            return SHORT_CONTEXT
        case "two_blocks":
            return LONG_CONTEXT
        case _:
            raise ValueError(f"Unsupported condition: {condition_name}")


def resolve_prompt(prompt: str | None, *, corpus: Corpus, condition_name: str) -> str:
    resolved_prompt = prompt if prompt is not None else corpus.val_text[:LONG_CONTEXT]
    required_context = prompt_context_size(condition_name)
    if len(resolved_prompt) < required_context:
        raise ValueError(
            f"Prompt must be at least {required_context} characters for {condition_name}, got {len(resolved_prompt)}"
        )
    return resolved_prompt


@torch.inference_mode()
def generate_text(
    model: TiedReadoutModel,
    *,
    prompt: str,
    corpus: Corpus,
    sample_temperature: float,
    device: torch.device,
    generated_characters: int = 500,
) -> str:
    model.eval()
    token_buffer = encode_prompt(prompt, stoi=corpus.stoi)

    normalized_embeddings = None
    if model.normalize:
        normalized_embeddings = F.normalize(model.token_embedding.weight.float(), dim=-1).to(model.token_embedding.weight.dtype)

    for _ in range(generated_characters):
        short_inputs = torch.tensor([token_buffer[-SHORT_CONTEXT:]], dtype=torch.long, device=device)
        mid_inputs = None
        long_inputs = None
        if model.mid_block is not None:
            mid_inputs = torch.tensor([token_buffer[-MID_CONTEXT:]], dtype=torch.long, device=device)
        if model.long_block is not None:
            long_inputs = torch.tensor([token_buffer[-LONG_CONTEXT:]], dtype=torch.long, device=device)

        with autocast_context(device):
            combined_hidden = model.block_last_hidden(model.output_block, short_inputs)
            if mid_inputs is not None and model.mid_block is not None:
                mid_hidden = model.block_last_hidden(model.mid_block, mid_inputs).detach()
                combined_hidden = combined_hidden + model.lateral_scale * mid_hidden
            if long_inputs is not None and model.long_block is not None:
                long_hidden = model.block_last_hidden(model.long_block, long_inputs).detach()
                combined_hidden = combined_hidden + model.lateral_scale * long_hidden

            if model.normalize:
                logits = F.linear(combined_hidden, normalized_embeddings) / model.temperature
            else:
                logits = F.linear(combined_hidden, model.token_embedding.weight) / model.temperature

        next_token = sample_from_logits(logits.squeeze(0), sample_temperature=sample_temperature)
        token_buffer.append(int(next_token.item()))

    return "".join(corpus.itos[token] for token in token_buffer[-generated_characters:])


def print_generated_text(*, condition_name: str, prompt: str, generated_text: str) -> None:
    print(f"\n[{condition_name}] prompt", flush=True)
    print(prompt, flush=True)
    print(f"\n[{condition_name}] generated_text", flush=True)
    print(generated_text, flush=True)


def warm_up_cuda(
    train_dataset: WindowDataset,
    *,
    device: torch.device,
    model_size: ModelSize,
    temperature: float,
    lateral_scale: float,
    normalize: bool,
    local_loss: str,
    batch_size: int,
) -> None:
    if device.type != "cuda":
        return

    set_seed(DEFAULT_SEED)
    model = build_model(
        "two_blocks",
        vocab_size=train_dataset.vocab_size,
        model_size=model_size,
        temperature=temperature,
        lateral_scale=lateral_scale,
        normalize=normalize,
        local_loss=local_loss,
    ).to(device)
    optimizer = make_optimizer(model, device)
    short_inputs, mid_inputs, long_inputs, targets = sample_batch(train_dataset, batch_size=batch_size, device=device)

    model.train()
    with autocast_context(device):
        mid_hidden = model.block_last_hidden_only(model.mid_block, mid_inputs)
        long_hidden = model.block_last_hidden_only(model.long_block, long_inputs)
        output_logits = model.output_logits(short_inputs, mid_hidden=mid_hidden, long_hidden=long_hidden)
        output_loss, _ = next_char_metrics(output_logits, targets)
        mid_loss = interior_local_loss(
            mid_hidden,
            embedding=model.token_embedding,
            targets=targets,
            temperature=model.temperature,
            normalize=model.normalize,
            local_loss=model.local_loss,
        )
        long_loss = interior_local_loss(
            long_hidden,
            embedding=model.token_embedding,
            targets=targets,
            temperature=model.temperature,
            normalize=model.normalize,
            local_loss=model.local_loss,
        )
        total_loss = output_loss + mid_loss + long_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode(), autocast_context(device):
        warm_mid_hidden = model.block_last_hidden(model.mid_block, mid_inputs)
        warm_long_hidden = model.block_last_hidden(model.long_block, long_inputs)
        model.output_logits(short_inputs, mid_hidden=warm_mid_hidden, long_hidden=warm_long_hidden)

    torch.cuda.synchronize()


@torch.no_grad()
def evaluate_condition(model: TiedReadoutModel, dataset: WindowDataset, *, batch_size: int, device: torch.device) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = dataset.targets.shape[0]

    for start in range(0, total_examples, batch_size):
        stop = min(start + batch_size, total_examples)
        short_inputs = dataset.short_inputs[start:stop]
        mid_inputs = dataset.mid_inputs[start:stop]
        long_inputs = dataset.long_inputs[start:stop]
        targets = dataset.targets[start:stop]

        with autocast_context(device):
            mid_hidden = model.block_last_hidden(model.mid_block, mid_inputs) if model.mid_block is not None else None
            long_hidden = model.block_last_hidden(model.long_block, long_inputs) if model.long_block is not None else None
            logits = model.output_logits(short_inputs, mid_hidden=mid_hidden, long_hidden=long_hidden)
            loss, _ = next_char_metrics(logits, targets)

        total_loss += loss.item() * (stop - start)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()

    return EvalResult(val_loss=total_loss / total_examples, val_accuracy=total_correct / total_examples)


def train_condition(
    condition_name: str,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    *,
    seed: int,
    device: torch.device,
    model_size: ModelSize,
    temperature: float,
    lateral_scale: float,
    normalize: bool,
    local_loss: str,
    training_steps: int,
    batch_size: int,
) -> TrainedCondition:
    set_seed(seed)
    model = build_model(
        condition_name,
        vocab_size=train_dataset.vocab_size,
        model_size=model_size,
        temperature=temperature,
        lateral_scale=lateral_scale,
        normalize=normalize,
        local_loss=local_loss,
    ).to(device)
    optimizer = make_optimizer(model, device)
    started_at = perf_counter()

    for step in range(1, training_steps + 1):
        model.train()
        short_inputs, mid_inputs, long_inputs, targets = sample_batch(train_dataset, batch_size=batch_size, device=device)

        with autocast_context(device):
            total_loss_terms: list[Tensor] = []
            mid_hidden = None
            long_hidden = None
            mid_loss_value = 0.0
            long_loss_value = 0.0

            if model.mid_block is not None:
                mid_hidden = model.block_last_hidden_only(model.mid_block, mid_inputs)
                mid_loss = interior_local_loss(
                    mid_hidden,
                    embedding=model.token_embedding,
                    targets=targets,
                    temperature=model.temperature,
                    normalize=model.normalize,
                    local_loss=model.local_loss,
                )
                total_loss_terms.append(mid_loss)
                mid_loss_value = mid_loss.item()

            if model.long_block is not None:
                long_hidden = model.block_last_hidden_only(model.long_block, long_inputs)
                long_loss = interior_local_loss(
                    long_hidden,
                    embedding=model.token_embedding,
                    targets=targets,
                    temperature=model.temperature,
                    normalize=model.normalize,
                    local_loss=model.local_loss,
                )
                total_loss_terms.append(long_loss)
                long_loss_value = long_loss.item()

            output_logits = model.output_logits(short_inputs, mid_hidden=mid_hidden, long_hidden=long_hidden)
            output_loss, output_accuracy = next_char_metrics(output_logits, targets)
            total_loss_terms.insert(0, output_loss)
            total_loss = sum(total_loss_terms)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % PRINT_INTERVAL == 0 or step == training_steps:
            print(
                f"[{condition_name}] step={step:04d}/{training_steps} "
                f"block0_ce={output_loss.item():.4f} block0_acc={output_accuracy.item():.4%} "
                f"mid_loss={mid_loss_value:.4f} long_loss={long_loss_value:.4f}",
                flush=True,
            )

    evaluation = evaluate_condition(model, val_dataset, batch_size=EVAL_BATCH_SIZE, device=device)
    wall_seconds = perf_counter() - started_at
    print(
        f"[{condition_name}] final val_loss={evaluation.val_loss:.4f} "
        f"val_accuracy={evaluation.val_accuracy:.4%} wall_seconds={wall_seconds:.2f}",
        flush=True,
    )
    return TrainedCondition(
        result=ConditionResult(
            name=condition_name,
            val_loss=evaluation.val_loss,
            val_accuracy=evaluation.val_accuracy,
            delta_from_baseline=0.0,
            wall_seconds=wall_seconds,
        ),
        model=model,
    )


def print_results_table(results: list[ConditionResult]) -> None:
    baseline_loss = next((result.val_loss for result in results if result.name == "block0_alone"), None)
    header = "condition".ljust(20) + "val_loss".rjust(12) + "val_acc".rjust(12) + "delta".rjust(12) + "wall_s".rjust(12)
    print("\nresults", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for result in results:
        delta = result.delta_from_baseline if baseline_loss is not None else float("nan")
        print(
            result.name.ljust(20)
            + f"{result.val_loss:12.4f}"
            + f"{result.val_accuracy:12.4%}"
            + f"{delta:12.4f}"
            + f"{result.wall_seconds:12.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    requested_conditions = parse_condition_names(args.conditions)
    model_size = resolve_model_size(args)
    device = resolve_device(args.device)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset, val_dataset, corpus = load_window_dataset()
    train_dataset = dataset_to_device(train_dataset, device)
    val_dataset = dataset_to_device(val_dataset, device)
    warm_up_cuda(
        train_dataset,
        device=device,
        model_size=model_size,
        temperature=args.temperature,
        lateral_scale=args.lateral_scale,
        normalize=args.normalize,
        local_loss=args.local_loss,
        batch_size=args.batch_size,
    )
    print(
        "tied readout lm "
        f"device={device.type} seed={args.seed} train_examples={train_dataset.targets.shape[0]} "
        f"val_examples={val_dataset.targets.shape[0]} vocab={train_dataset.vocab_size} "
        f"short_ctx={SHORT_CONTEXT} mid_ctx={MID_CONTEXT} long_ctx={LONG_CONTEXT} "
        f"d_model={model_size.d_model} n_heads={model_size.n_heads} ff_dim={model_size.ff_dim} n_layers={model_size.n_layers} "
        f"steps={args.steps} batch={args.batch_size} temperature={args.temperature:.4f} lateral_scale={args.lateral_scale:.4f} normalize={args.normalize} local_loss={args.local_loss} "
        f"generate={args.generate} sample_temperature={args.sample_temperature:.4f} "
        f"conditions={','.join(requested_conditions)}",
        flush=True,
    )

    trained_conditions = [
        train_condition(
            condition_name,
            train_dataset,
            val_dataset,
            seed=args.seed,
            device=device,
            model_size=model_size,
            temperature=args.temperature,
            lateral_scale=args.lateral_scale,
            normalize=args.normalize,
            local_loss=args.local_loss,
            training_steps=args.steps,
            batch_size=args.batch_size,
        )
        for condition_name in requested_conditions
    ]

    for trained_condition in trained_conditions:
        if not args.generate:
            continue
        prompt = resolve_prompt(args.prompt, corpus=corpus, condition_name=trained_condition.result.name)
        generated_text = generate_text(
            trained_condition.model,
            prompt=prompt,
            corpus=corpus,
            sample_temperature=args.sample_temperature,
            device=device,
        )
        print_generated_text(
            condition_name=trained_condition.result.name,
            prompt=prompt,
            generated_text=generated_text,
        )

    results = [trained_condition.result for trained_condition in trained_conditions]

    baseline_loss = next((result.val_loss for result in results if result.name == "block0_alone"), None)
    adjusted_results = [
        ConditionResult(
            name=result.name,
            val_loss=result.val_loss,
            val_accuracy=result.val_accuracy,
            delta_from_baseline=(result.val_loss - baseline_loss) if baseline_loss is not None else float("nan"),
            wall_seconds=result.wall_seconds,
        )
        for result in results
    ]
    print_results_table(adjusted_results)


if __name__ == "__main__":
    main()
