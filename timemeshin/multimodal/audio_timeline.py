import torch
import torch.nn as nn

class TimeMeshinAudioTimelineEncoder(nn.Module):
    """
    TimeMeshin Audio Timeline Encoder:
    Ingests continuous audio spectrograms or raw 1D audio waveforms.
    Extracts acoustic onset energy peaks as I-Frames and tonal/spectral envelopes as B-Frames.
    """
    def __init__(self, in_features=80, dim=128, onset_energy_threshold=0.45):
        super().__init__()
        self.in_features = in_features
        self.dim = dim
        self.onset_energy_threshold = onset_energy_threshold

        # 1D Temporal Convolution to downsample fine audio timesteps to frame-level rate
        self.audio_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(dim)
        )

    def forward(self, mel_spectrogram: torch.Tensor):
        """
        Args:
            mel_spectrogram: Tensor of shape (batch, n_mels, time_steps)
        Returns:
            audio_embeddings: Tensor of shape (batch, downsampled_time, dim)
            timeline_metadata: List of frame tags ('I-FRAME' or 'B-FRAME')
        """
        # (batch, dim, downsampled_time)
        # Apply conv layers
        x = self.audio_conv[0](mel_spectrogram)
        x = self.audio_conv[1](x)
        x = self.audio_conv[2](x)
        
        # Permute for LayerNorm: (batch, time, dim)
        x = x.transpose(1, 2)
        audio_embeddings = self.audio_conv[3](x)

        batch_size, time_steps, _ = audio_embeddings.shape
        metadata = []
        prev_vector = None

        for t in range(time_steps):
            curr_vector = audio_embeddings[:, t, :]
            if prev_vector is None:
                tag = "I-FRAME"  # Audio utterance onset anchor
            else:
                # Calculate acoustic onset derivative
                onset_delta = torch.norm(curr_vector - prev_vector, dim=-1).mean().item()
                tag = "I-FRAME" if onset_delta > self.onset_energy_threshold else "B-FRAME"

            prev_vector = curr_vector.detach()
            metadata.append(tag)

        return audio_embeddings, metadata
