import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _rvq_lookup_kernel(
        X_ptr, C_ptr, Out_ptr, Ind_ptr,
        M, N, K,
        stride_xm, stride_xn,
        stride_cm, stride_ck,
        stride_om, stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < M

        x_offsets = rows[:, None] * stride_xm + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_xn
        x_vals = tl.load(X_ptr + x_offsets, mask=row_mask[:, None], other=0.0)

        best_dist = tl.full((BLOCK_SIZE_M,), 1e9, dtype=tl.float32)
        best_idx = tl.full((BLOCK_SIZE_M,), -1, dtype=tl.int32)

        for k_idx in range(0, K):
            c_offsets = k_idx * stride_cm + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ck
            c_vals = tl.load(C_ptr + c_offsets)

            diff = x_vals - c_vals
            dist = tl.sum(diff * diff, axis=1)
            updated_mask = dist < best_dist
            best_dist = tl.where(updated_mask, dist, best_dist)
            best_idx = tl.where(updated_mask, k_idx, best_idx)

        tl.store(Ind_ptr + rows, best_idx, mask=row_mask)

def triton_rvq_snapping(x: torch.Tensor, codebook_weight: torch.Tensor):
    """
    GPU-accelerated RVQ codebook snapping using Triton (or PyTorch fallback).
    """
    if not HAS_TRITON or not x.is_cuda:
        # Fallback to PyTorch native vectorized cdist
        distances = torch.cdist(x.unsqueeze(1), codebook_weight.unsqueeze(0)).squeeze(1)
        indices = torch.argmin(distances, dim=-1)
        quantized = codebook_weight[indices.long()]
        return x + (quantized - x).detach(), indices

    M, N = x.shape
    K, _ = codebook_weight.shape
    out = torch.empty_like(x)
    indices = torch.empty((M,), dtype=torch.int32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    _rvq_lookup_kernel[grid](
        x, codebook_weight, out, indices,
        M, N, K,
        x.stride(0), x.stride(1),
        codebook_weight.stride(0), codebook_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=N
    )
    quantized = codebook_weight[indices.long()]
    return x + (quantized - x).detach(), indices
