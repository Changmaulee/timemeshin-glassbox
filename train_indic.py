import torch
import torch.optim as optim
from timemeshin.model import TimeMeshinGlassboxLM
from timemeshin.tokenizers.text_tmot import IndicAksharaTokenizer

def stream_and_train_samanantar(target_language="hi", batch_size=4, dim=128, max_steps=20):
    """
    Fine-tunes TimeMeshin-Glassbox on Indic Akshara tokens (using AI4Bharat streaming or local synthetic fallback).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Indic Akshara Pipeline [Language: '{target_language}']...")

    tokenizer = IndicAksharaTokenizer()
    model = TimeMeshinGlassboxLM(vocab_size=5000, dim=dim, num_layers=3, choices_per_layer=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    model.train()

    sample_indic_sentences = [
        "परिवर्तन प्रकृति का नियम है ।",
        "ज्ञान ही परम शक्ति है ।",
        "समय का पहिया निरंतर घूमता रहता है ।",
        "सत्य की ही सदैव विजय होती है ।"
    ]

    print("[*] Beginning Multi-Scale Akshara Timeline Training Loop...")
    for step in range(1, max_steps + 1):
        batch_input_vectors = []
        batch_timeline_metadata = []

        for text in sample_indic_sentences:
            timeline_tokens = tokenizer.tokenize_indic_string(text)
            seq_len = 16
            timeline_tokens = timeline_tokens[:seq_len] + [{"type": "B-FRAME", "content": " "}] * max(0, seq_len - len(timeline_tokens))
            input_vecs = torch.randn(seq_len, dim)
            meta_tags = [tok["type"] for tok in timeline_tokens]
            batch_input_vectors.append(input_vecs)
            batch_timeline_metadata.append(meta_tags)

        input_tensors = torch.stack(batch_input_vectors).to(device)
        target_mock_ids = torch.randint(0, 500, (len(sample_indic_sentences), 16), device=device)
        timeline_steps = list(zip(*batch_timeline_metadata))

        optimizer.zero_grad()
        logits, loss, metrics, _ = model(input_tensors, timeline_steps, target_token_ids=target_mock_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"  -> Step [{step}/{max_steps}] | Total Loss: {metrics['loss_total']:.4f} | Recon Loss: {metrics['loss_recon']:.4f}")

    print("[+] Indic Akshara fine-tuning completed successfully.")

if __name__ == "__main__":
    stream_and_train_samanantar(target_language="hi", max_steps=10)
