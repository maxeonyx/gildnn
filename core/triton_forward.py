from __future__ import annotations

import torch
import triton
import triton.language as tl
from einops import rearrange
from torch import Tensor

from core.automaton_graph import GraphCellularAutomaton, l2_normalize


TILE_B = 16
TILE_H = 64
BLOCK_D = 128
EPSILON = 1e-6


@triton.jit
def _gelu_approx(x: tl.tensor) -> tl.tensor:
    inv_sqrt2 = 0.7071067811865476
    return 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))


@triton.jit
def automaton_timestep_kernel(
    prev_state_ptr,
    prev_global_ptr,
    next_state_ptr,
    next_global_ptr,
    prev_pred_ptr,
    next_pred_ptr,
    token_embedding_ptr,
    noise_ptr,
    fires_ptr,
    band0_mask_ptr,
    neighbor_indices_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    pred_w_ptr,
    pred_b_ptr,
    batch_size,
    prev_state_stride_m,
    prev_state_stride_b,
    prev_state_stride_d,
    prev_global_stride_m,
    prev_global_stride_b,
    prev_global_stride_d,
    next_state_stride_m,
    next_state_stride_b,
    next_state_stride_d,
    next_global_stride_m,
    next_global_stride_b,
    next_global_stride_d,
    prev_pred_stride_m,
    prev_pred_stride_b,
    prev_pred_stride_d,
    next_pred_stride_m,
    next_pred_stride_b,
    next_pred_stride_d,
    token_stride_b,
    token_stride_d,
    noise_stride_m,
    noise_stride_b,
    noise_stride_d,
    w1_stride_m,
    w1_stride_in,
    w1_stride_out,
    b1_stride_m,
    b1_stride_out,
    w2_stride_m,
    w2_stride_in,
    w2_stride_out,
    b2_stride_m,
    b2_stride_out,
    pred_w_stride_m,
    pred_w_stride_in,
    pred_w_stride_out,
    pred_b_stride_m,
    pred_b_stride_out,
    D_STREAM: tl.constexpr,
    D_HIDDEN: tl.constexpr,
    TILE_B: tl.constexpr,
    TILE_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    module_idx = tl.program_id(0)
    batch_tile_idx = tl.program_id(1)

    batch_offsets = batch_tile_idx * TILE_B + tl.arange(0, TILE_B)
    d_offsets = tl.arange(0, BLOCK_D)
    batch_mask = batch_offsets < batch_size
    d_mask = d_offsets < D_STREAM

    prev_state_ptrs = (
        prev_state_ptr
        + module_idx * prev_state_stride_m
        + batch_offsets[:, None] * prev_state_stride_b
        + d_offsets[None, :] * prev_state_stride_d
    )
    prev_global_ptrs = (
        prev_global_ptr
        + module_idx * prev_global_stride_m
        + batch_offsets[:, None] * prev_global_stride_b
        + d_offsets[None, :] * prev_global_stride_d
    )
    prev_pred_ptrs = (
        prev_pred_ptr
        + module_idx * prev_pred_stride_m
        + batch_offsets[:, None] * prev_pred_stride_b
        + d_offsets[None, :] * prev_pred_stride_d
    )

    prev_state = tl.load(prev_state_ptrs, mask=batch_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    prev_global = tl.load(prev_global_ptrs, mask=batch_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    prev_pred = tl.load(prev_pred_ptrs, mask=batch_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    neighbor_sum = tl.zeros((TILE_B, BLOCK_D), dtype=tl.float32)
    for neighbor_slot in range(4):
        neighbor_idx = tl.load(neighbor_indices_ptr + module_idx * 4 + neighbor_slot)
        neighbor_valid = neighbor_idx >= 0
        safe_neighbor_idx = tl.where(neighbor_valid, neighbor_idx, 0)
        neighbor_ptrs = (
            prev_global_ptr
            + safe_neighbor_idx * prev_global_stride_m
            + batch_offsets[:, None] * prev_global_stride_b
            + d_offsets[None, :] * prev_global_stride_d
        )
        neighbor_sum += tl.load(
            neighbor_ptrs,
            mask=batch_mask[:, None] & d_mask[None, :] & neighbor_valid,
            other=0.0,
        ).to(tl.float32)

    noise_ptrs = (
        noise_ptr
        + module_idx * noise_stride_m
        + batch_offsets[:, None] * noise_stride_b
        + d_offsets[None, :] * noise_stride_d
    )
    combined = prev_state + neighbor_sum + tl.load(
        noise_ptrs,
        mask=batch_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    token_ptrs = token_embedding_ptr + batch_offsets[:, None] * token_stride_b + d_offsets[None, :] * token_stride_d
    band0_mask = tl.load(band0_mask_ptr + module_idx).to(tl.float32)
    combined += tl.load(token_ptrs, mask=batch_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32) * band0_mask

    norm = tl.sqrt(tl.sum(combined * combined, axis=1) + 1e-6)
    normalized = combined / norm[:, None]

    output = tl.zeros((TILE_B, BLOCK_D), dtype=tl.float32)
    for hidden_start in range(0, D_HIDDEN, TILE_H):
        hidden_offsets = hidden_start + tl.arange(0, TILE_H)
        hidden_mask = hidden_offsets < D_HIDDEN
        w1_ptrs = (
            w1_ptr
            + module_idx * w1_stride_m
            + d_offsets[:, None] * w1_stride_in
            + hidden_offsets[None, :] * w1_stride_out
        )
        b1_ptrs = b1_ptr + module_idx * b1_stride_m + hidden_offsets * b1_stride_out
        w1 = tl.load(w1_ptrs, mask=d_mask[:, None] & hidden_mask[None, :], other=0.0).to(tl.float32)
        b1 = tl.load(b1_ptrs, mask=hidden_mask, other=0.0).to(tl.float32)
        hidden = tl.dot(normalized, w1) + b1[None, :]
        activated = _gelu_approx(hidden)

        w2_ptrs = (
            w2_ptr
            + module_idx * w2_stride_m
            + hidden_offsets[:, None] * w2_stride_in
            + d_offsets[None, :] * w2_stride_out
        )
        w2 = tl.load(w2_ptrs, mask=hidden_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        output += tl.dot(activated, w2)

    b2_ptrs = b2_ptr + module_idx * b2_stride_m + d_offsets * b2_stride_out
    output += tl.load(b2_ptrs, mask=d_mask, other=0.0).to(tl.float32)[None, :]

    pred_w_ptrs = (
        pred_w_ptr
        + module_idx * pred_w_stride_m
        + d_offsets[:, None] * pred_w_stride_in
        + d_offsets[None, :] * pred_w_stride_out
    )
    pred_b_ptrs = pred_b_ptr + module_idx * pred_b_stride_m + d_offsets * pred_b_stride_out
    pred_w = tl.load(pred_w_ptrs, mask=d_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    pred = tl.dot(output, pred_w)
    pred += tl.load(pred_b_ptrs, mask=d_mask, other=0.0).to(tl.float32)[None, :]

    fire = tl.load(fires_ptr + module_idx) != 0
    next_state = tl.where(fire, output, prev_state)
    next_global = tl.where(fire, output, prev_global)
    next_pred = tl.where(fire, pred, prev_pred)

    next_state_ptrs = (
        next_state_ptr
        + module_idx * next_state_stride_m
        + batch_offsets[:, None] * next_state_stride_b
        + d_offsets[None, :] * next_state_stride_d
    )
    next_global_ptrs = (
        next_global_ptr
        + module_idx * next_global_stride_m
        + batch_offsets[:, None] * next_global_stride_b
        + d_offsets[None, :] * next_global_stride_d
    )
    next_pred_ptrs = (
        next_pred_ptr
        + module_idx * next_pred_stride_m
        + batch_offsets[:, None] * next_pred_stride_b
        + d_offsets[None, :] * next_pred_stride_d
    )
    tl.store(next_state_ptrs, next_state, mask=batch_mask[:, None] & d_mask[None, :])
    tl.store(next_global_ptrs, next_global, mask=batch_mask[:, None] & d_mask[None, :])
    tl.store(next_pred_ptrs, next_pred, mask=batch_mask[:, None] & d_mask[None, :])


def _validate_triton_inputs(
    model: GraphCellularAutomaton,
    tokens: Tensor,
    states: Tensor,
    global_buffer: Tensor,
    predictions: Tensor,
    has_predicted: Tensor,
    refractory_levels: Tensor,
) -> None:
    if tokens.ndim != 2:
        raise ValueError(f"Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}.")
    if tokens.dtype != torch.long:
        raise ValueError(f"Expected tokens dtype torch.long, got {tokens.dtype}.")
    if tokens.device.type != "cuda":
        raise ValueError(f"Triton forward requires CUDA tensors, got device {tokens.device}.")
    if model.refractory:
        raise ValueError("Triton v1 does not support refractory dynamics.")
    if model.d_stream != 96:
        raise ValueError(f"Triton v1 expects d_stream=96, got {model.d_stream}.")
    if model.d_hidden != 384:
        raise ValueError(f"Triton v1 expects d_hidden=384, got {model.d_hidden}.")
    expected_state_shape = (model.n_modules, tokens.shape[0], model.d_stream)
    for name, tensor in (("states", states), ("global_buffer", global_buffer), ("predictions", predictions)):
        if tensor.shape != expected_state_shape:
            raise ValueError(f"Expected {name} shape {expected_state_shape}, got {tuple(tensor.shape)}.")
        if tensor.device != tokens.device:
            raise ValueError(f"Expected {name} on {tokens.device}, got {tensor.device}.")
    if has_predicted.shape != (model.n_modules,):
        raise ValueError(f"Expected has_predicted shape {(model.n_modules,)}, got {tuple(has_predicted.shape)}.")
    if refractory_levels.shape != (model.n_modules,):
        raise ValueError(f"Expected refractory_levels shape {(model.n_modules,)}, got {tuple(refractory_levels.shape)}.")


def _pad_module_batch_tensor(tensor: Tensor, padded_batch: int) -> Tensor:
    if tensor.shape[1] == padded_batch:
        return tensor.contiguous()
    padded = torch.zeros(
        (tensor.shape[0], padded_batch, tensor.shape[2]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[:, : tensor.shape[1]] = tensor
    return padded.contiguous()


def _pad_batch_tensor(tensor: Tensor, padded_batch: int) -> Tensor:
    if tensor.shape[0] == padded_batch:
        return tensor.contiguous()
    padded = torch.zeros(
        (padded_batch, tensor.shape[1]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[: tensor.shape[0]] = tensor
    return padded.contiguous()


def _neighbor_sum_and_noise(
    model: GraphCellularAutomaton,
    current_buffer: Tensor,
) -> tuple[Tensor, Tensor]:
    gathered = current_buffer.detach()[model.clamped_neighbor_indices]
    masked_neighbors = gathered * model.neighbor_mask.to(device=current_buffer.device, dtype=current_buffer.dtype)
    if model.noise_std == 0.0:
        aggregated_noise = torch.zeros_like(masked_neighbors[:, 0])
    else:
        raw_noise = torch.randn_like(gathered) * model.noise_std
        aggregated_noise = (
            raw_noise * model.neighbor_mask.to(device=current_buffer.device, dtype=current_buffer.dtype)
        ).sum(dim=1)
    neighbor_sum = masked_neighbors.sum(dim=1)
    return neighbor_sum, aggregated_noise


def triton_forward_chunk(
    model: GraphCellularAutomaton,
    tokens: Tensor,
    states: Tensor,
    global_buffer: Tensor,
    predictions: Tensor,
    has_predicted: Tensor,
    refractory_levels: Tensor,
    global_step_offset: int | Tensor,
) -> tuple[Tensor, Tensor | None, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _validate_triton_inputs(model, tokens, states, global_buffer, predictions, has_predicted, refractory_levels)

    batch_size, seq_len = tokens.shape
    padded_batch = max(TILE_B, triton.cdiv(batch_size, TILE_B) * TILE_B)
    token_embeddings = model.token_embedding(tokens)
    current_states = _pad_module_batch_tensor(states, padded_batch)
    current_global_buffer = _pad_module_batch_tensor(global_buffer, padded_batch)
    current_predictions = _pad_module_batch_tensor(predictions, padded_batch)
    next_states = torch.empty_like(current_states)
    next_global_buffer = torch.empty_like(current_global_buffer)
    next_predictions = torch.empty_like(current_predictions)

    logits = torch.empty((seq_len, batch_size, model.vocab_size), device=tokens.device, dtype=token_embeddings.dtype)
    prediction_loss_sums = torch.zeros((model.n_modules,), device=tokens.device, dtype=token_embeddings.dtype)
    prediction_counts = torch.zeros((model.n_modules,), device=tokens.device, dtype=torch.long)
    current_has_predicted = has_predicted.clone()
    current_refractory_levels = refractory_levels.clone()

    if isinstance(global_step_offset, int):
        step_offset = torch.tensor(global_step_offset, device=tokens.device, dtype=torch.long)
    else:
        step_offset = global_step_offset.to(device=tokens.device, dtype=torch.long)

    total_steps = seq_len * model.steps_per_token
    local_step_offsets = torch.arange(total_steps, device=tokens.device, dtype=torch.long)
    fires_at = (
        torch.remainder(
            step_offset + local_step_offsets[:, None] + model.module_phases[None, :],
            model.module_rates[None, :],
        )
        == 0
    )

    band0_mask = model.band0_mask.to(device=tokens.device, dtype=torch.int32).contiguous()
    neighbor_indices = model.neighbor_indices.to(device=tokens.device, dtype=torch.int32).contiguous()

    for timestep in range(total_steps):
        token_index = timestep // model.steps_per_token
        fires_bool = fires_at[timestep]
        fires = fires_bool.to(dtype=torch.int32).contiguous()
        neighbor_sum, aggregated_noise = _neighbor_sum_and_noise(model, current_global_buffer)
        noisy_sum = neighbor_sum + aggregated_noise

        prediction_target = l2_normalize(noisy_sum[:, :batch_size])
        active_predictions = fires_bool & current_has_predicted
        prediction_errors = model._prediction_errors(current_predictions[:, :batch_size], prediction_target.detach())
        prediction_errors[model.band0_mask] = 0.0
        prediction_loss_sums = prediction_loss_sums + (
            prediction_errors * active_predictions.to(dtype=prediction_errors.dtype)
        )
        prediction_counts = prediction_counts + active_predictions.to(dtype=torch.long)
        non_band0_active = active_predictions.clone()
        non_band0_active[model.band0_mask] = False
        model._update_contrastive_buffer(prediction_target.detach(), non_band0_active)

        padded_token = _pad_batch_tensor(token_embeddings[:, token_index, :], padded_batch)
        padded_noise = _pad_module_batch_tensor(aggregated_noise, padded_batch)
        grid = (model.n_modules, triton.cdiv(padded_batch, TILE_B))
        automaton_timestep_kernel[grid](
            current_states,
            current_global_buffer,
            next_states,
            next_global_buffer,
            current_predictions,
            next_predictions,
            padded_token,
            padded_noise,
            fires,
            band0_mask,
            neighbor_indices,
            model.w1,
            model.b1,
            model.w2,
            model.b2,
            model.pred_w,
            model.pred_b,
            padded_batch,
            current_states.stride(0),
            current_states.stride(1),
            current_states.stride(2),
            current_global_buffer.stride(0),
            current_global_buffer.stride(1),
            current_global_buffer.stride(2),
            next_states.stride(0),
            next_states.stride(1),
            next_states.stride(2),
            next_global_buffer.stride(0),
            next_global_buffer.stride(1),
            next_global_buffer.stride(2),
            current_predictions.stride(0),
            current_predictions.stride(1),
            current_predictions.stride(2),
            next_predictions.stride(0),
            next_predictions.stride(1),
            next_predictions.stride(2),
            padded_token.stride(0),
            padded_token.stride(1),
            padded_noise.stride(0),
            padded_noise.stride(1),
            padded_noise.stride(2),
            model.w1.stride(0),
            model.w1.stride(1),
            model.w1.stride(2),
            model.b1.stride(0),
            model.b1.stride(2),
            model.w2.stride(0),
            model.w2.stride(1),
            model.w2.stride(2),
            model.b2.stride(0),
            model.b2.stride(2),
            model.pred_w.stride(0),
            model.pred_w.stride(1),
            model.pred_w.stride(2),
            model.pred_b.stride(0),
            model.pred_b.stride(2),
            D_STREAM=model.d_stream,
            D_HIDDEN=model.d_hidden,
            TILE_B=TILE_B,
            TILE_H=TILE_H,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )

        current_states, next_states = next_states, current_states
        current_global_buffer, next_global_buffer = next_global_buffer, current_global_buffer
        current_predictions, next_predictions = next_predictions, current_predictions
        current_has_predicted = current_has_predicted | fires_bool

        if timestep % model.steps_per_token == model.steps_per_token - 1:
            band0_logits = model.logits_from_hidden(current_states[model.band0_mask, :batch_size])
            logits[token_index] = band0_logits.mean(dim=0)

    return (
        rearrange(logits, "seq batch vocab -> batch seq vocab"),
        None,
        current_states[:, :batch_size],
        current_global_buffer[:, :batch_size],
        current_predictions[:, :batch_size],
        current_has_predicted,
        current_refractory_levels,
        prediction_loss_sums,
        prediction_counts,
    )


class TritonForwardChunkFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        model: GraphCellularAutomaton,
        tokens: Tensor,
        states: Tensor,
        global_buffer: Tensor,
        predictions: Tensor,
        has_predicted: Tensor,
        refractory_levels: Tensor,
        global_step_offset: int | Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        del ctx
        return triton_forward_chunk(
            model,
            tokens,
            states,
            global_buffer,
            predictions,
            has_predicted,
            refractory_levels,
            global_step_offset,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: Tensor,
    ) -> tuple[None, None, None, None, None, None, None, None, None]:
        del ctx, grad_outputs
        raise NotImplementedError("Triton backward is not implemented yet.")
