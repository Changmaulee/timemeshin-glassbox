# ==============================================================================
# TIMEMESHIN-GLASSBOX: RESILIENT GPU PRE-TRAINING ENGINE (GOOGLE COLAB / RUNPOD)
# Features: Auto-Resume, Google Drive Checkpoint Persistence, Mixed Precision (AMP)
# ==============================================================================

import os
import sys
import glob
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset

# Determine Best Persistent Checkpoint Directory (Google Drive or Local)
DRIVE_DIR = "/content/drive/MyDrive/timemeshin_checkpoints"
LOCAL_DIR = "./checkpoints"
CHECKPOINT_DIR = DRIVE_DIR if os.path.exists("/content/drive/MyDrive") else LOCAL_DIR
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CONFIG = {
    "model_name": "TimeMeshin-Indic-125M",
    "tokenizer_id": "changmaulee/timemeshin-indic-otm-tokenizer",
    "dim": 512,                  # Latent dimension (512 for ~125M scaled model)
    "num_rvq_layers": 4,         # 4-stage Hierarchical RVQ Abacus
    "choices_per_layer": 1024,   # 1024 discrete centroids per layer
    "seq_len": 256,              # Context window length
    "batch_size": 16,            # Micro-batch size per GPU
    "grad_accum_steps": 4,       # Effective batch size = 16 * 4 = 64
    "learning_rate": 4e-4,       # Peak learning rate
    "warmup_steps": 500,
    "max_steps": 10000,          # Total pre-training steps
    "save_interval": 250,        # Save checkpoint every 250 steps
    "checkpoint_dir": CHECKPOINT_DIR,
    "mixed_precision": True      # FP16 / BF16 AMP for 2.5x speedup
}

class ScaledTimeMeshinGlassboxLM(nn.Module):
    def __init__(self, vocab_size, dim=512, num_rvq_layers=4, choices_per_layer=1024):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_rvq_layers = num_rvq_layers

        # Token Embedding
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # Layer 1 & 2: Hierarchical RVQ Abacus Codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(choices_per_layer, dim) for _ in range(num_rvq_layers)
        ])

        # Layer 4: Multi-Scale Timescale Decay Rates (Macro tau=0.95, Delta tau=0.40)
        self.register_buffer("tau_macro", torch.tensor(0.95))
        self.register_buffer("tau_delta", torch.tensor(0.40))
        self.transition_matrix = nn.Linear(dim, dim, bias=False)

        # Layer 5: Gated Auditable Glassbox FFN
        self.glass_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False)
        )
        self.norm = nn.LayerNorm(dim)

        # Layer 6: Output Language Modeling Head (Tied Weights)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed input tokens
        x = self.tok_embeddings(input_ids)

        # Hierarchical RVQ Vectorized Snapping across all tokens
        flat_x = x.reshape(-1, self.dim)
        residual = flat_x
        quantized_total = torch.zeros_like(flat_x)
        rvq_commitment_loss = 0.0

        for cb in self.codebooks:
            dists = torch.cdist(residual.unsqueeze(1), cb.weight.unsqueeze(0)).squeeze(1)
            indices = torch.argmin(dists, dim=-1)
            q_layer = cb(indices)

            # Straight-Through Estimator (STE)
            q_ste = residual + (q_layer - residual).detach()
            quantized_total = quantized_total + q_ste

            rvq_commitment_loss += torch.mean((residual - q_layer.detach()) ** 2)
            residual = residual - q_layer

        quant_seq = quantized_total.reshape(batch_size, seq_len, self.dim)

        # Layer 4: Dual-Clock Linear State Space Recurrence
        states = []
        running_h = torch.zeros(batch_size, self.dim, device=device)

        for t in range(seq_len):
            tau = self.tau_macro if t % 4 == 0 else self.tau_delta
            h_next = tau * running_h + (1.0 - tau) * self.transition_matrix(quant_seq[:, t, :])
            running_h = h_next
            states.append(running_h)

        h_stacked = torch.stack(states, dim=1)
        h_norm = self.norm(h_stacked)

        # Layer 5: Feedforward pass
        features = self.glass_ffn(h_norm) + h_norm

        # Layer 6: Output Logits
        logits = self.lm_head(features)
        return logits, rvq_commitment_loss / self.num_rvq_layers

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def find_latest_checkpoint(ckpt_dir):
    checkpoints = glob.glob(os.path.join(ckpt_dir, "timemeshin_step_*.pt"))
    if not checkpoints:
        return None
    # Sort by step index
    checkpoints.sort(key=lambda x: int(x.split("_step_")[-1].replace(".pt", "")))
    return checkpoints[-1]

def train_gpu_scaled():
    print("=========================================================================================")
    print("    TIMEMESHIN-GLASSBOX: RESILIENT GPU PRE-TRAINING ENGINE (WITH AUTO-RESUME)            ")
    print("=========================================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[+] Active GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[!] CUDA not detected; running in CPU mode.")

    print(f"[+] Checkpoint Storage Path: {CONFIG['checkpoint_dir']}")

    # Load Tokenizer
    print(f"[*] Loading Tokenizer: {CONFIG['tokenizer_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_id"])
    vocab_size = len(tokenizer)
    print(f"[+] Tokenizer Vocab Size: {vocab_size:,} tokens")

    # Initialize Model & Optimizer
    model = ScaledTimeMeshinGlassboxLM(
        vocab_size=vocab_size,
        dim=CONFIG["dim"],
        num_rvq_layers=CONFIG["num_rvq_layers"],
        choices_per_layer=CONFIG["choices_per_layer"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=CONFIG["mixed_precision"] and device.type == "cuda")

    # 1. Check for Existing Checkpoint to Resume
    start_step = 0
    latest_ckpt = find_latest_checkpoint(CONFIG["checkpoint_dir"])

    if latest_ckpt:
        print(f"\n[🔄 RESUME DETECTED] Found existing checkpoint: {latest_ckpt}")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        if "scaler_state" in ckpt_data and device.type == "cuda":
            scaler.load_state_dict(ckpt_data["scaler_state"])
        start_step = ckpt_data.get("step", 0)
        print(f"[+] Successfully resumed from Step {start_step:,} / {CONFIG['max_steps']:,}!\n")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[+] Fresh Training Initialization: {total_params:,} parameters (~{total_params/1e6:.1f}M)\n")

    # Connect to Streaming Multilingual Data
    print("[*] Connecting to Streaming Multilingual Corpus...")
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
        data_iter = iter(dataset)
        print("[+] Live Hugging Face dataset stream active!")
    except Exception as e:
        print(f"[?] Notice: Online stream fallback to local high-density Indic buffer: {e}")
        data_iter = None

    fallback_corpus = [
        "ज्ञान और विद्या मनुष्य का सबसे बड़ा धन है । परिवर्तन ही संसार का शाश्वत नियम है ।",
        "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः ।",
        "கற்க கசடறக் கற்றவை கற்றபின் நிற்க அதற்குத் தக. யாதும் ஊரே யாவரும் கேளிர்.",
        "దేశభాషలందు తెలుగు లెస్స అని శ్రీకృష్ణదేవరాయలు కీర్తించిరి.",
        "ಸಿರಿಗನ್ನಡಂ ಗೆಲ್ಗೆ ಸಿರಿಗನ್ನಡಂ ಬಾಳ್ಗೆ ಎಂದು ಕನ್ನಡದ ಕವಿಗಳು ಹಾಡಿದ್ದಾರೆ.",
        "মোদের গরব মোদের আশা আমরি বাংলা ভাষা । চিত্ত যেথা ভয়শূন্য उच्च যেথা শির ।"
    ]

    model.train()
    step = start_step
    accum_tokens = 0
    start_time = time.time()

    while step < CONFIG["max_steps"]:
        step += 1
        lr = get_lr(step, CONFIG["warmup_steps"], CONFIG["max_steps"], CONFIG["learning_rate"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(CONFIG["grad_accum_steps"]):
            batch_texts = []
            for _ in range(CONFIG["batch_size"]):
                if data_iter:
                    try:
                        sample = next(data_iter)
                        batch_texts.append(sample.get("text", "")[:500])
                    except StopIteration:
                        data_iter = iter(dataset)
                        sample = next(data_iter)
                        batch_texts.append(sample.get("text", "")[:500])
                else:
                    batch_texts.append(fallback_corpus[(step + _) % len(fallback_corpus)])

            enc = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=CONFIG["seq_len"],
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            targets = input_ids.clone()

            with autocast(enabled=CONFIG["mixed_precision"] and device.type == "cuda"):
                logits, rvq_loss = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()

                ce_loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_targets.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                total_loss = (ce_loss + 0.1 * rvq_loss) / CONFIG["grad_accum_steps"]

            scaler.scale(total_loss).backward()
            loss_accum += total_loss.item()
            accum_tokens += CONFIG["batch_size"] * CONFIG["seq_len"]

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Telemetry
        if step % 20 == 0 or step == start_step + 1:
            elapsed = time.time() - start_time
            tok_per_sec = accum_tokens / max(1e-5, elapsed)
            print(f"  [Step {step:05d}/{CONFIG['max_steps']}] | Loss: {loss_accum*CONFIG['grad_accum_steps']:.4f} | CE: {ce_loss.item():.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.1f} tok/s")

        # Save Checkpoint Directly to Drive / Local
        if step % CONFIG["save_interval"] == 0 or step == CONFIG["max_steps"]:
            save_path = os.path.join(CONFIG["checkpoint_dir"], f"timemeshin_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": CONFIG
            }, save_path)
            print(f"  [💾 CHECKPOINT SAVED] => {save_path}")

    print(f"\n[+] Pre-training completed in {(time.time()-start_time)/60:.2f} minutes!")

if __name__ == "__main__":
    train_gpu_scaled()
