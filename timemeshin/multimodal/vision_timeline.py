import torch
import torch.nn as nn

class TimeMeshinVisionPatchEncoder(nn.Module):
    """
    TimeMeshin Vision Patch Timeline Encoder:
    Converts 2D visual frames / patches into deterministic I-Frame & B-Frame sequences.
    Detects visual novelty / scene shifts as I-Frames and temporal background persistence as B-Frames.
    """
    def __init__(self, in_channels=3, patch_size=16, dim=128, novelty_threshold=0.35):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.novelty_threshold = novelty_threshold

        # Lightweight convolutional patch projection (no heavy ViT attention required)
        self.patch_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, video_or_image_tensor: torch.Tensor):
        """
        Args:
            video_or_image_tensor: Tensor of shape (batch, frames, channels, height, width) 
                                   or (batch, channels, height, width) for static image.
        Returns:
            patch_embeddings: Tensor of shape (batch, total_patches, dim)
            timeline_metadata: List of frame tags ('I-FRAME' or 'B-FRAME')
        """
        if video_or_image_tensor.dim() == 4:
            # Static image: (batch, C, H, W) -> add time/frame dimension (batch, 1, C, H, W)
            video_or_image_tensor = video_or_image_tensor.unsqueeze(1)

        batch_size, num_frames, c, h, w = video_or_image_tensor.shape
        all_embeddings = []
        all_metadata = []

        prev_patch_rep = None

        for f in range(num_frames):
            frame = video_or_image_tensor[:, f, :, :, :]  # (batch, C, H, W)
            # Patch projection
            patches = self.patch_conv(frame)  # (batch, dim, H/patch, W/patch)
            patches = patches.flatten(2).transpose(1, 2)  # (batch, num_patches, dim)
            patches = self.norm(patches)

            num_patches = patches.shape[1]
            for p in range(num_patches):
                curr_patch = patches[:, p, :]
                if prev_patch_rep is None:
                    tag = "I-FRAME"  # First frame / patch anchor
                else:
                    delta_energy = torch.norm(curr_patch - prev_patch_rep, dim=-1).mean().item()
                    tag = "I-FRAME" if delta_energy > self.novelty_threshold else "B-FRAME"

                prev_patch_rep = curr_patch.detach()
                all_embeddings.append(curr_patch)
                all_metadata.append(tag)

        # (batch, total_sequence_length, dim)
        stacked_embeddings = torch.stack(all_embeddings, dim=1)
        return stacked_embeddings, all_metadata
