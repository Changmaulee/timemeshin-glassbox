import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from timemeshin.model import TimeMeshinGlassboxLM

class MultiModalTimeMeshinDataset(Dataset):
    def __init__(self, datatype="text", num_samples=100, seq_len=32, dim=256):
        self.datatype = datatype
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dim = dim
        self.data_vectors = torch.randn(num_samples, seq_len, dim)
        self.target_tokens = torch.randint(0, 1000, (num_samples, seq_len))
        self.timeline_metadata = []

        for _ in range(num_samples):
            if datatype == "text":
                meta = ["I-FRAME" if i % 8 == 0 else "B-FRAME" for i in range(seq_len)]
            elif datatype == "code":
                meta = ["I-FRAME" if i % 16 == 0 else "B-FRAME" for i in range(seq_len)]
            else:
                meta = ["I-FRAME" if i % 4 == 0 else "B-FRAME" for i in range(seq_len)]
            self.timeline_metadata.append(meta)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_vectors[idx], self.target_tokens[idx], self.timeline_metadata[idx]

def run_production_training_loop(modality_type="text", epochs=2):
    print(f"\n[*] Setting up Glassbox Training Node | Modality: {modality_type.upper()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MultiModalTimeMeshinDataset(datatype=modality_type, num_samples=64, seq_len=16, dim=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TimeMeshinGlassboxLM(vocab_size=1000, dim=128, num_layers=3, choices_per_layer=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(1, epochs + 1):
        total_epoch_loss = 0.0
        for vectors, targets, metadata_batch in dataloader:
            vectors, targets = vectors.to(device), targets.to(device)
            timeline_steps = list(zip(*metadata_batch))

            optimizer.zero_grad()
            logits, loss, metrics, _ = model(vectors, timeline_steps, target_token_ids=targets)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(dataloader)
        print(f"  -> Epoch {epoch} Completed | Loss Metrics Summary: {avg_loss:.4f}")

if __name__ == "__main__":
    run_production_training_loop(modality_type="text", epochs=2)
    run_production_training_loop(modality_type="code", epochs=2)
    run_production_training_loop(modality_type="sensor", epochs=2)
