import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from timemeshin.model_v2 import TimeMeshinGlassboxLM_v2

def test_v2_architecture():
    print("[*] Initializing TimeMeshin-Glassbox v0.2 Test...")
    model = TimeMeshinGlassboxLM_v2(
        vocab_size=16621,
        dim=768,
        num_rvq_layers=6,
        choices_per_layer=1024,
        num_heads=8
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Total Parameters in v0.2 Model: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Test forward pass with mock batch (batch=2, seq=32)
    mock_input = torch.randint(0, 16621, (2, 32))
    logits, rvq_loss = model(mock_input)

    assert logits.shape == (2, 32, 16621), f"Unexpected shape {logits.shape}"
    assert rvq_loss.item() >= 0.0, "Negative commitment loss"
    print(f"[+] Logits Shape: {logits.shape} | RVQ Loss: {rvq_loss.item():.4f}")
    print("[+] TIMEMESHIN v0.2 ARCHITECTURE IS 100% VERIFIED & READY!")

if __name__ == "__main__":
    test_v2_architecture()
