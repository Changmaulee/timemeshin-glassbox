import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from timemeshin.multimodal.multimodal_fused_lm import TimeMeshinOmniLM
from timemeshin.multimodal.vision_timeline import TimeMeshinVisionPatchEncoder
from timemeshin.multimodal.audio_timeline import TimeMeshinAudioTimelineEncoder

def test_vision_patch_timeline():
    encoder = TimeMeshinVisionPatchEncoder(in_channels=3, patch_size=16, dim=64)
    # Mock video: 2 frames of 32x32 image (4 patches per frame = 8 patches total)
    mock_video = torch.randn(2, 2, 3, 32, 32)
    embeddings, meta = encoder(mock_video)

    assert embeddings.shape == (2, 8, 64)
    assert len(meta) == 8
    assert meta[0] == "I-FRAME"  # First frame must be an I-FRAME anchor

def test_audio_timeline_encoder():
    encoder = TimeMeshinAudioTimelineEncoder(in_features=80, dim=64)
    # Mock Mel spectrogram: 80 mels across 16 timesteps (downsampled by 4 = 4 steps)
    mock_mels = torch.randn(2, 80, 16)
    embeddings, meta = encoder(mock_mels)

    assert embeddings.shape == (2, 4, 64)
    assert len(meta) == 4
    assert meta[0] == "I-FRAME"

def test_omni_modal_fused_forward():
    model = TimeMeshinOmniLM(vocab_size=1000, dim=64, num_layers=2, choices_per_layer=64)
    
    mock_text = torch.randint(0, 1000, (2, 6))
    mock_image = torch.randn(2, 3, 32, 32) # 4 patches
    mock_audio = torch.randn(2, 80, 16)    # 4 acoustic frames

    logits, seq, meta, ledger = model.forward_multimodal(
        text_token_ids=mock_text,
        video_or_images=mock_image,
        audio_mels=mock_audio
    )

    # Total timesteps = 4 (vision) + 4 (audio) + 6 (text) = 14 steps
    assert seq.shape == (2, 14, 64)
    assert logits.shape == (2, 14, 1000)
    assert len(meta) == 14
    print("[+] OmniLM Multi-Modal Forward Passed: Fused Vision (4) + Audio (4) + Text (6) = 14 Timesteps!")

if __name__ == "__main__":
    test_vision_patch_timeline()
    print("[+] test_vision_patch_timeline PASSED")
    test_audio_timeline_encoder()
    print("[+] test_audio_timeline_encoder PASSED")
    test_omni_modal_fused_forward()
    print("[+] test_omni_modal_fused_forward PASSED")
    print("\n[+] ALL MULTI-MODAL TIMELINE TESTS 100% PASSING!")
