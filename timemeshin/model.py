import torch
import torch.nn as nn
from timemeshin.layers.timeline import TimeMeshinTimelineEngine
from timemeshin.loss import HierarchicalRVQLoss

class TimeMeshinGlassboxLM(nn.Module):
    """
    TimeMeshin-Glassbox Language Model:
    Complete 6-layer inherently interpretable sequence architecture.
    Bypasses quadratic multi-head attention via Hierarchical RVQ and Multi-Scale Timescale State Spaces.
    """
    def __init__(self, vocab_size: int = 10000, dim: int = 256, num_layers: int = 3, choices_per_layer: int = 256):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Layers 1 & 2: Hierarchical Abacus Codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(choices_per_layer, dim) for _ in range(num_layers)
        ])

        # Layer 4: Multi-Scale Timeline Engine
        self.timeline = TimeMeshinTimelineEngine(dim=dim)

        # Layer 5: Fully Traceable Auditable Glassbox FFN
        self.glass_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(dim * 2, dim, bias=False)
        )

        # Layer 6: Output Vocabulary Head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Loss Optimization Engine
        self.loss_engine = HierarchicalRVQLoss()

    def quantization_pass(self, raw_embeddings: torch.Tensor):
        # Flatten batch and sequence dimensions for fast vectorized parallel quantization
        orig_shape = raw_embeddings.shape
        flat_input = raw_embeddings.reshape(-1, self.dim)
        residual = flat_input
        quantized_total = torch.zeros_like(flat_input)
        residual_records = []
        selected_embeddings = []

        for codebook in self.codebooks:
            residual_records.append(residual.clone())
            distances = torch.cdist(residual.unsqueeze(1), codebook.weight.unsqueeze(0)).squeeze(1)
            indices = torch.argmin(distances, dim=-1)
            quantized_layer = codebook(indices)

            # Straight-Through Estimator (STE)
            quantized_layer_ste = residual + (quantized_layer - residual).detach()
            quantized_total = quantized_total + quantized_layer_ste
            selected_embeddings.append(quantized_layer)
            residual = residual - quantized_layer

        quantized_reshaped = quantized_total.reshape(orig_shape)
        return quantized_reshaped, residual_records, selected_embeddings

    def forward(self, input_vectors: torch.Tensor, metadata_timeline: list, target_token_ids: torch.Tensor = None):
        """
        Executes an end-to-end forward trace pass through all 6 Glassbox layers.
        """
        batch_size, sequence_length, _ = input_vectors.shape
        running_state = torch.zeros(batch_size, self.dim, device=input_vectors.device)
        ledger_traces = []

        # Layer 2: Vectorized parallel codebook snapping across full sequence
        quant_vectors, residuals, selected = self.quantization_pass(input_vectors)

        # Layer 4: High-speed temporal recurrence
        all_states = []
        for t in range(sequence_length):
            if isinstance(metadata_timeline[t], (list, tuple)):
                step_meta_tag = metadata_timeline[t][0]
            else:
                step_meta_tag = metadata_timeline[t]
            frame_meta = {"type": step_meta_tag}
            current_quant = quant_vectors[:, t, :]

            running_state = self.timeline.process_step(frame_meta, current_quant, t, running_state)
            all_states.append(running_state)

            if sequence_length <= 32:  # Detailed tracing for short inspection sequences
                ledger_traces.append({
                    "timestep": t,
                    "frame_type": step_meta_tag,
                    "state_vector_sample": running_state[0, :min(4, self.dim)].tolist()
                })

        stacked_states = torch.stack(all_states, dim=1)

        # Layer 5: Vectorized FFN Pass
        ffn_features = self.glass_ffn(stacked_states)

        # Layer 6: Output Head Logits
        logits = self.output_head(ffn_features)

        if target_token_ids is not None:
            flat_input = input_vectors.reshape(-1, self.dim)
            total_loss, metrics = self.loss_engine(flat_input, residuals, selected)
            return logits, total_loss, metrics, ledger_traces

        return logits, ledger_traces
