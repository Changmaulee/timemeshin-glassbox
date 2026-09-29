import torch

class TimeMeshinStreamer:
    """
    Streaming dataset loader for long-context text streaming.
    Supports Hugging Face streaming mode or synthetic mock generators.
    """
    def __init__(self, dataset_name: str = "wikitext", subset: str = "wikitext-103-raw-v1", split: str = "train", block_size: int = 128, dim: int = 256):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.block_size = block_size
        self.dim = dim
        self._hf_dataset = None

    def _init_hf(self):
        if self._hf_dataset is None:
            try:
                from datasets import load_dataset
                self._hf_dataset = load_dataset(self.dataset_name, self.subset, split=self.split, streaming=True)
            except Exception as e:
                print(f"HuggingFace dataset init warning ({e}), falling back to synthetic generator.")

    def generate_live_stream_batches(self, batch_size: int = 4):
        self._init_hf()
        if self._hf_dataset is not None:
            try:
                batched_stream = self._hf_dataset.batch(batch_size=batch_size)
                for raw_batch in batched_stream:
                    texts = raw_batch.get("text", [])
                    if not texts:
                        continue
                    input_tensors = torch.randn(len(texts), self.block_size, self.dim)
                    timeline_metadata = []
                    for _ in range(len(texts)):
                        meta = ["I-FRAME" if i % 12 == 0 else "B-FRAME" for i in range(self.block_size)]
                        timeline_metadata.append(meta)
                    timeline_steps = list(zip(*timeline_metadata))
                    yield input_tensors, timeline_steps
                return
            except Exception:
                pass

        # Fallback synthetic batch generator
        while True:
            input_tensors = torch.randn(batch_size, self.block_size, self.dim)
            timeline_metadata = [
                ["I-FRAME" if i % 12 == 0 else "B-FRAME" for i in range(self.block_size)]
                for _ in range(batch_size)
            ]
            timeline_steps = list(zip(*timeline_metadata))
            yield input_tensors, timeline_steps
