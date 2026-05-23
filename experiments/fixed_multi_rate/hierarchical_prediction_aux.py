from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.model import ParallelDiagonalForwardState, ParallelDiagonalModel


@dataclass
class AuxLossTensors:
    task_loss: Float[Tensor, ""]
    probe_loss: Float[Tensor, ""]
    hier_loss: Float[Tensor, ""]
    total_loss: Float[Tensor, ""]
    per_block_probe_losses: Float[Tensor, "num_blocks"]
    per_pair_hier_losses: Float[Tensor, "num_pairs"]
    per_block_latent_dim_var: Float[Tensor, "num_blocks"]


def next_token_targets(
    tokens: Int[Tensor, "batch context"],
    final_targets: Int[Tensor, "batch"],
) -> Int[Tensor, "batch context"]:
    return torch.cat([tokens[:, 1:], final_targets[:, None]], dim=1)


class HierarchicalPredictionAux(nn.Module):
    def __init__(
        self,
        *,
        context_size: int,
        d_model: int,
        d_z: int,
        vocab_size: int,
        rates: tuple[int, ...],
        include_hierarchical: bool,
    ) -> None:
        super().__init__()
        if len(rates) < 2:
            raise ValueError("HierarchicalPredictionAux requires at least two block rates.")
        self.context_size = context_size
        self.d_model = d_model
        self.d_z = d_z
        self.vocab_size = vocab_size
        self.rates = tuple(rates)
        self.include_hierarchical = include_hierarchical
        self.num_blocks = len(rates)
        self.num_pairs = len(rates) - 1

        self.latent_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in rates])
        self.latent_projections = nn.ModuleList([nn.Linear(d_model, d_z) for _ in rates])
        self.probe_heads = nn.ModuleList([nn.Linear(d_z, vocab_size) for _ in rates])

        pair_ratios: list[int] = []
        pair_masks: list[Tensor] = []
        time_index = torch.arange(context_size, dtype=torch.long)
        for fast_rate, slow_rate in zip(rates[:-1], rates[1:], strict=True):
            if slow_rate % fast_rate != 0:
                raise ValueError(
                    f"Adjacent rates must divide cleanly for hierarchical prediction, got {fast_rate} -> {slow_rate}."
                )
            ratio = slow_rate // fast_rate
            pair_ratios.append(ratio)
            pair_masks.append(((time_index % slow_rate) == 0) & (time_index + (ratio * fast_rate) < context_size))
        self.pair_ratios = tuple(pair_ratios)
        self.register_buffer("pair_masks", torch.stack(pair_masks).to(torch.float32), persistent=False)

        if include_hierarchical:
            self.predictor_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_pairs)])
            self.predictor_heads = nn.ModuleList(
                [nn.Linear(d_model, ratio * d_z) for ratio in self.pair_ratios]
            )
        else:
            self.predictor_norms = nn.ModuleList()
            self.predictor_heads = nn.ModuleList()

    def _compute_latents(
        self,
        block_outputs: list[Float[Tensor, "batch context d_model"]],
    ) -> tuple[list[Float[Tensor, "batch context d_z"]], Float[Tensor, "num_blocks"]]:
        latents: list[Tensor] = []
        latent_dim_vars: list[Tensor] = []
        for block_output, latent_norm, latent_projection in zip(
            block_outputs,
            self.latent_norms,
            self.latent_projections,
            strict=True,
        ):
            latent_pre = latent_projection(latent_norm(block_output))
            latents.append(F.normalize(latent_pre, dim=-1, eps=1e-6))
            latent_dim_vars.append(latent_pre.var(dim=(0, 1), unbiased=False).mean())
        return latents, torch.stack(latent_dim_vars)

    def compute_losses(
        self,
        *,
        state: ParallelDiagonalForwardState,
        tokens: Int[Tensor, "batch context"],
        final_targets: Int[Tensor, "batch"],
        final_logits: Float[Tensor, "batch vocab"],
        lambda_probe: float,
        lambda_hier: float,
        warmup_scale: Float[Tensor, ""],
    ) -> AuxLossTensors:
        task_loss = F.cross_entropy(final_logits, final_targets)
        per_timestep_targets = next_token_targets(tokens, final_targets)
        block_outputs = state.block_outputs
        latents, per_block_latent_dim_var = self._compute_latents(block_outputs)

        probe_losses = [
            F.cross_entropy(
                rearrange(probe_head(latent), "batch context vocab -> (batch context) vocab"),
                rearrange(per_timestep_targets, "batch context -> (batch context)"),
            )
            for latent, probe_head in zip(latents, self.probe_heads, strict=True)
        ]
        per_block_probe_losses = torch.stack(probe_losses)
        probe_loss = per_block_probe_losses.mean()

        if self.include_hierarchical:
            pair_losses: list[Tensor] = []
            batch_size = tokens.shape[0]
            for pair_index, ratio in enumerate(self.pair_ratios):
                slow_index = pair_index + 1
                fast_index = pair_index
                predictor_input = self.predictor_norms[pair_index](block_outputs[slow_index])
                predicted = self.predictor_heads[pair_index](predictor_input)
                predicted = rearrange(
                    predicted,
                    "batch context (horizon d_z) -> batch context horizon d_z",
                    horizon=ratio,
                    d_z=self.d_z,
                )
                target = torch.stack(
                    [
                        torch.roll(latents[fast_index], shifts=-(horizon * self.rates[fast_index]), dims=1)
                        for horizon in range(1, ratio + 1)
                    ],
                    dim=2,
                )
                cosine_distance = 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1, eps=1e-6)
                pair_mask = self.pair_masks[pair_index].view(1, self.context_size, 1).expand(batch_size, -1, ratio)
                pair_losses.append((cosine_distance * pair_mask).sum() / pair_mask.sum().clamp_min(1.0))
            per_pair_hier_losses = torch.stack(pair_losses)
            hier_loss = per_pair_hier_losses.mean()
        else:
            per_pair_hier_losses = final_logits.new_zeros((self.num_pairs,))
            hier_loss = final_logits.new_zeros(())

        total_loss = task_loss + warmup_scale * ((lambda_probe * probe_loss) + (lambda_hier * hier_loss))
        return AuxLossTensors(
            task_loss=task_loss,
            probe_loss=probe_loss,
            hier_loss=hier_loss,
            total_loss=total_loss,
            per_block_probe_losses=per_block_probe_losses,
            per_pair_hier_losses=per_pair_hier_losses,
            per_block_latent_dim_var=per_block_latent_dim_var,
        )


class HierarchicalPredictionGraphTrainer:
    def __init__(
        self,
        *,
        model: ParallelDiagonalModel,
        aux_module: HierarchicalPredictionAux,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        lambda_probe: float,
        lambda_hier: float,
        aux_warmup_steps: int,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"HierarchicalPredictionGraphTrainer requires CUDA, got {device}.")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError(f"Expected AdamW, got {type(optimizer).__name__}.")
        self.model = model
        self.aux_module = aux_module
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.lambda_probe = lambda_probe
        self.lambda_hier = lambda_hier
        self.aux_warmup_steps = aux_warmup_steps
        self.capture_stream = torch.cuda.Stream(device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.is_captured = False

        self.static_input = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
        self.static_target = torch.empty((batch_size,), device=device, dtype=torch.long)
        self.static_step = torch.zeros((), device=device, dtype=torch.float32)
        self.static_task_loss = torch.zeros((), device=device)
        self.static_probe_loss = torch.zeros((), device=device)
        self.static_hier_loss = torch.zeros((), device=device)
        self.static_total_loss = torch.zeros((), device=device)
        self.static_per_block_probe_losses = torch.zeros((aux_module.num_blocks,), device=device)
        self.static_per_pair_hier_losses = torch.zeros((aux_module.num_pairs,), device=device)
        self.static_per_block_latent_dim_var = torch.zeros((aux_module.num_blocks,), device=device)

    def _copy_batch(
        self,
        *,
        batch_input: Int[Tensor, "batch context"],
        batch_target: Int[Tensor, "batch"],
        step: int,
    ) -> None:
        if tuple(batch_input.shape) != (self.batch_size, self.seq_len):
            raise ValueError(f"Expected input shape {(self.batch_size, self.seq_len)}, got {tuple(batch_input.shape)}.")
        if tuple(batch_target.shape) != (self.batch_size,):
            raise ValueError(f"Expected target shape {(self.batch_size,)}, got {tuple(batch_target.shape)}.")
        self.static_input.copy_(batch_input, non_blocking=True)
        self.static_target.copy_(batch_target, non_blocking=True)
        self.static_step.fill_(float(step))

    def _training_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        logits, state = self.model.forward_with_state(self.static_input)
        warmup_scale = torch.clamp(self.static_step / float(self.aux_warmup_steps), max=1.0)
        losses = self.aux_module.compute_losses(
            state=state,
            tokens=self.static_input,
            final_targets=self.static_target,
            final_logits=logits,
            lambda_probe=self.lambda_probe,
            lambda_hier=self.lambda_hier,
            warmup_scale=warmup_scale,
        )
        losses.total_loss.backward()
        self.optimizer.step()
        self.static_task_loss.copy_(losses.task_loss.detach())
        self.static_probe_loss.copy_(losses.probe_loss.detach())
        self.static_hier_loss.copy_(losses.hier_loss.detach())
        self.static_total_loss.copy_(losses.total_loss.detach())
        self.static_per_block_probe_losses.copy_(losses.per_block_probe_losses.detach())
        self.static_per_pair_hier_losses.copy_(losses.per_pair_hier_losses.detach())
        self.static_per_block_latent_dim_var.copy_(losses.per_block_latent_dim_var.detach())

    def capture(
        self,
        warmup_batches: list[tuple[Int[Tensor, "batch context"], Int[Tensor, "batch"], int]],
    ) -> None:
        if self.is_captured:
            raise RuntimeError("capture() may only be called once.")
        if len(warmup_batches) < 3:
            raise ValueError(f"Need at least 3 warmup batches, got {len(warmup_batches)}.")

        self.model.train()
        self.aux_module.train()
        current_stream = torch.cuda.current_stream(device=self.device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            for batch_input, batch_target, step in warmup_batches[:3]:
                self._copy_batch(batch_input=batch_input, batch_target=batch_target, step=step)
                self._training_step()
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(self.device)

        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._training_step()

        current_stream.wait_stream(self.capture_stream)
        self.is_captured = True

    def step(
        self,
        *,
        batch_input: Int[Tensor, "batch context"],
        batch_target: Int[Tensor, "batch"],
        step: int,
    ) -> AuxLossTensors:
        if not self.is_captured:
            raise RuntimeError("step() requires capture() first.")
        self._copy_batch(batch_input=batch_input, batch_target=batch_target, step=step)
        self.graph.replay()
        return self.snapshot()

    def snapshot(self) -> AuxLossTensors:
        return AuxLossTensors(
            task_loss=self.static_task_loss.detach().clone(),
            probe_loss=self.static_probe_loss.detach().clone(),
            hier_loss=self.static_hier_loss.detach().clone(),
            total_loss=self.static_total_loss.detach().clone(),
            per_block_probe_losses=self.static_per_block_probe_losses.detach().clone(),
            per_pair_hier_losses=self.static_per_pair_hier_losses.detach().clone(),
            per_block_latent_dim_var=self.static_per_block_latent_dim_var.detach().clone(),
        )

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)


@torch.inference_mode()
def evaluate_hierarchical_prediction(
    *,
    model: ParallelDiagonalModel,
    aux_module: HierarchicalPredictionAux | None,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    batch_size: int,
) -> dict[str, float | list[float]]:
    was_training_model = model.training
    was_training_aux = aux_module.training if aux_module is not None else False
    model.eval()
    if aux_module is not None:
        aux_module.eval()

    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    total_probe_loss = 0.0
    total_hier_loss = 0.0
    total_per_block_probe_losses = None
    total_per_pair_hier_losses = None
    total_per_block_latent_dim_var = None

    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        batch_examples = batch_targets.shape[0]
        logits, state = model.forward_with_state(batch_inputs)
        total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
        total_examples += batch_examples

        if aux_module is None:
            continue

        aux_losses = aux_module.compute_losses(
            state=state,
            tokens=batch_inputs,
            final_targets=batch_targets,
            final_logits=logits,
            lambda_probe=0.0,
            lambda_hier=0.0,
            warmup_scale=torch.ones((), device=batch_inputs.device, dtype=torch.float32),
        )
        total_probe_loss += aux_losses.probe_loss.item() * batch_examples
        total_hier_loss += aux_losses.hier_loss.item() * batch_examples

        if total_per_block_probe_losses is None:
            total_per_block_probe_losses = aux_losses.per_block_probe_losses.detach().clone() * batch_examples
            total_per_pair_hier_losses = aux_losses.per_pair_hier_losses.detach().clone() * batch_examples
            total_per_block_latent_dim_var = aux_losses.per_block_latent_dim_var.detach().clone() * batch_examples
        else:
            total_per_block_probe_losses += aux_losses.per_block_probe_losses.detach() * batch_examples
            total_per_pair_hier_losses += aux_losses.per_pair_hier_losses.detach() * batch_examples
            total_per_block_latent_dim_var += aux_losses.per_block_latent_dim_var.detach() * batch_examples

    if was_training_model:
        model.train()
    if aux_module is not None and was_training_aux:
        aux_module.train()

    if aux_module is None:
        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
            "probe_loss": 0.0,
            "hier_loss": 0.0,
            "per_block_probe_losses": [0.0] * model.num_blocks,
            "per_pair_hier_losses": [0.0] * (model.num_blocks - 1),
            "per_block_latent_dim_var": [0.0] * model.num_blocks,
        }

    if total_per_block_probe_losses is None or total_per_pair_hier_losses is None or total_per_block_latent_dim_var is None:
        raise RuntimeError("Auxiliary metrics were not accumulated.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "probe_loss": total_probe_loss / total_examples,
        "hier_loss": total_hier_loss / total_examples,
        "per_block_probe_losses": (total_per_block_probe_losses / total_examples).cpu().tolist(),
        "per_pair_hier_losses": (total_per_pair_hier_losses / total_examples).cpu().tolist(),
        "per_block_latent_dim_var": (total_per_block_latent_dim_var / total_examples).cpu().tolist(),
    }
