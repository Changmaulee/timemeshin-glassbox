import torch
import torch.nn as nn

class HierarchicalRVQLoss(nn.Module):
    """
    Hierarchical Residual Vector Quantization (RVQ) Loss Function.
    Calculates reconstruction loss along with dictionary and commitment loss terms
    using straight-through estimator (stop-gradient) formulations.
    """
    def __init__(self, beta_commitment: float = 0.25, beta_dictionary: float = 1.0):
        super().__init__()
        self.beta_c = beta_commitment
        self.beta_d = beta_dictionary

    def forward(self, original_input: torch.Tensor, residual_records: list, selected_embeddings: list):
        """
        Args:
            original_input: Tensor of shape [Batch, Dim]
            residual_records: List of Tensors, length L, shape [Batch, Dim] (residual before each layer)
            selected_embeddings: List of Tensors, length L, shape [Batch, Dim] (selected codebook beads)
        Returns:
            total_loss: Scalar optimization objective
            metrics: Dictionary of individual loss components
        """
        total_quantization_loss = 0.0
        reconstructed_output = torch.zeros_like(original_input)

        for r_prev, e_l in zip(residual_records, selected_embeddings):
            # 1. Dictionary Loss: Pulls codebook entries toward continuous data vectors
            loss_dict = torch.mean((r_prev.detach() - e_l) ** 2)
            # 2. Commitment Loss: Prevents encoder outputs from fluctuating wildly
            loss_commit = torch.mean((r_prev - e_l.detach()) ** 2)

            total_quantization_loss += (self.beta_d * loss_dict) + (self.beta_c * loss_commit)
            reconstructed_output += e_l

        # 3. Structural Reconstruction Loss
        reconstruction_loss = torch.mean((original_input - reconstructed_output) ** 2)
        total_loss = reconstruction_loss + total_quantization_loss

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_recon": reconstruction_loss.item(),
            "loss_quant": total_quantization_loss.item()
        }
