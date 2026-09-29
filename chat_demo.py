"""
Conversational English Word Generation Demo for TimeMeshin-Glassbox
Trains directly on conversational English sentence structures for 30 seconds (RAM < 25MB)
and demonstrates live natural language completion and Glassbox bead tracing.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from timemeshin.model import TimeMeshinGlassboxLM

# Curated foundational English corpus
TRAINING_SENTENCES = [
    "the cat sits on the warm mat",
    "the dog runs in the green park",
    "the sun shines bright in the blue sky",
    "a smart detective solves the hidden mystery",
    "knowledge and science bring true wisdom to humanity",
    "time moves forward and never stops",
    "the brave hero saves the peaceful city",
    "reading books expands our creative imagination",
    "water flows gently down the quiet river",
    "the golden bird sings a sweet song in the morning",
    "the quick brown fox jumps over the lazy dog",
    "artificial intelligence helps humans learn and grow",
    "truth always wins in the end of the journey",
    "the teacher explains the lesson clearly to the students",
    "a bright idea can change the whole world"
]

def train_and_speak_english(epochs: int = 150):
    print("=========================================================================================")
    print("      TIMEMESHIN-GLASSBOX CONVERSATIONAL ENGLISH WORD TRAINING & GENERATION DEMO        ")
    print("=========================================================================================\n")

    # 1. Build discrete English word dictionary
    all_words = set()
    for s in TRAINING_SENTENCES:
        for w in s.split():
            all_words.add(w)

    vocab = {word: idx for idx, word in enumerate(sorted(list(all_words)))}
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    vocab_size = len(vocab)
    dim = 64

    print(f"[*] English Vocabulary Size: {vocab_size} unique words")
    print(f"[*] Initializing 6-Layer TimeMeshin Architecture (Dim: {dim}, RAM: < 20 MB)...")

    model = TimeMeshinGlassboxLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=3,
        choices_per_layer=32
    )
    token_embed = nn.Embedding(vocab_size, dim)
    optimizer = optim.AdamW(list(model.parameters()) + list(token_embed.parameters()), lr=3e-3)

    # Convert sentences to padded tensors
    max_len = max(len(s.split()) for s in TRAINING_SENTENCES)
    sentence_tensors = []
    metadata_list = []

    for s in TRAINING_SENTENCES:
        words = s.split()
        ids = [vocab[w] for w in words]
        padded = ids + [0] * (max_len - len(ids))
        sentence_tensors.append(padded)
        meta = ["I-FRAME" if i == 0 or i == len(words)-1 else "B-FRAME" for i in range(max_len)]
        metadata_list.append(meta)

    input_data = torch.tensor(sentence_tensors)
    timeline_steps = list(zip(*metadata_list))

    print(f"[*] Training on sentence structures for {epochs} fast epochs...")
    start_time = time.time()
    model.train()

    for epoch in range(1, epochs + 1):
        input_vecs = token_embed(input_data)
        optimizer.zero_grad()

        # Parallel sequence forward pass
        logits, rvq_loss, metrics, _ = model(input_vecs, timeline_steps, target_token_ids=input_data)

        # Autoregressive target matching (predict next word)
        targets = input_data[:, 1:]
        pred_logits = logits[:, :-1, :]
        ce_loss = nn.functional.cross_entropy(pred_logits.reshape(-1, vocab_size), targets.reshape(-1))

        loss = ce_loss + (0.3 * rvq_loss)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0 or epoch == 1:
            print(f"  -> Epoch [{epoch:03d}/{epochs}] | Cross-Entropy Loss: {ce_loss.item():.4f} | Recon Loss: {metrics['loss_recon']:.4f}")

    train_time = time.time() - start_time
    print(f"\n[+] English sentence patterns learned in {train_time:.2f} seconds!\n")

    # -----------------------------------------------------------------
    # LIVE GENERATION TEST
    # -----------------------------------------------------------------
    print("=========================================================================================")
    print("                           LIVE ENGLISH SPEAK & COMPLETION TEST                          ")
    print("=========================================================================================")

    model.eval()
    test_prompts = [
        "the cat sits on",
        "the dog runs in",
        "a smart detective solves",
        "knowledge and science bring"
    ]

    with torch.no_grad():
        for prompt in test_prompts:
            words = prompt.split()
            ids = [vocab[w] for w in words if w in vocab]
            prompt_tensor = torch.tensor([ids])
            prompt_vecs = token_embed(prompt_tensor)
            prompt_meta = ["I-FRAME" if i == 0 else "B-FRAME" for i in range(len(ids))]

            logits, traces = model(prompt_vecs, prompt_meta)

            # Predict next words
            next_word_id = torch.argmax(logits[0, -1, :]).item()
            predicted_word = reverse_vocab.get(next_word_id, "[unknown]")

            print(f"\n[?] Prompt:    '{prompt} ...'")
            print(f"[!] AI Speaks: '{prompt} {predicted_word}'")
            print(f"[*] Glassbox Decision Trail: Step {len(ids)-1} [{traces[-1]['frame_type']}] -> Codebook Coordinate Sample: {traces[-1]['state_vector_sample']}")

    print("\n=========================================================================================")

if __name__ == "__main__":
    train_and_speak_english(epochs=150)
