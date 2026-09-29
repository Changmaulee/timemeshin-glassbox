import torch
import torch.nn as nn
from timemeshin.model import TimeMeshinGlassboxLM
from timemeshin.multimodal.vision_timeline import TimeMeshinVisionPatchEncoder
from timemeshin.multimodal.audio_timeline import TimeMeshinAudioTimelineEncoder

class TimeMeshinOmniLM(nn.Module):
    """
    TimeMeshin-OmniLM:
    Multi-Modal Glassbox State-Space Architecture.
    Interleaves Vision patches, Audio spectrogram frames, and Text Akshara tokens
    on a single synchronized temporal state timeline with Hierarchical RVQ.
    """
    def __init__(self, vocab_size=15923, dim=128, num_layers=3, choices_per_layer=512):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Modality Encoders
        self.text_embedder = nn.Embedding(vocab_size, dim)
        self.vision_encoder = TimeMeshinVisionPatchEncoder(dim=dim)
        self.audio_encoder = TimeMeshinAudioTimelineEncoder(dim=dim)

        # Core 6-Layer TimeMeshin Glassbox Engine
        self.glassbox_engine = TimeMeshinGlassboxLM(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            choices_per_layer=choices_per_layer
        )

    def forward_multimodal(self, text_token_ids=None, video_or_images=None, audio_mels=None):
        """
        Fuses multimodal streams chronologically onto the deterministic Glassbox timeline.
        """
        fused_vectors = []
        fused_timeline = []

        # 1. Process Vision Stream
        if video_or_images is not None:
            v_vecs, v_meta = self.vision_encoder(video_or_images)
            fused_vectors.append(v_vecs)
            fused_timeline.extend(v_meta)

        # 2. Process Audio Stream
        if audio_mels is not None:
            a_vecs, a_meta = self.audio_encoder(audio_mels)
            fused_vectors.append(a_vecs)
            fused_timeline.extend(a_meta)

        # 3. Process Text / Akshara Stream
        if text_token_ids is not None:
            t_vecs = self.text_embedder(text_token_ids)
            batch_size, seq_len, _ = t_vecs.shape
            t_meta = ["I-FRAME" if i % 4 == 0 else "B-FRAME" for i in range(seq_len)]
            fused_vectors.append(t_vecs)
            fused_timeline.extend(t_meta)

        if not fused_vectors:
            raise ValueError("Must provide at least one modality (text, vision, or audio).")

        # Concatenate across sequence dimension (batch, total_fused_timesteps, dim)
        full_sequence = torch.cat(fused_vectors, dim=1)

        # Pass through Glassbox Engine
        logits, ledger = self.glassbox_engine(full_sequence, fused_timeline)
        return logits, full_sequence, fused_timeline, ledger
