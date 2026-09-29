# TimeMeshin-Glassbox

A 6-Layer Inherently Interpretable Hierarchical State Space Language Model Architecture.

## Architecture Overview

```
[Input Stream] 
  ──► [Layer 1: Structural TMOT Tokenizer] (AST / Akshara Syllable I/B-Frames)
  ──► [Layer 2: Structural Embedding Abacus] (Hierarchical RVQ Codebooks)
  ──► [Layer 3: Positional Anchor] (Keyframe-relative)
  ──► [Layer 4: Multi-Scale State Space Timeline] (Dual-clock SSM + Flight Recorder)
  ──► [Layer 5: Gated Glassbox FFN] (Transparent Projection)
  ──► [Layer 6: Explainable Softmax Head & Audit Ledger]
```

## Repository Structure

- `timemeshin/`
  - `model.py`: Complete 6-layer model implementation.
  - `loss.py`: Hierarchical Residual Vector Quantization (RVQ) loss engine.
  - `layers/timeline.py`: Multi-Scale State Space Engine with I/B-Frame dynamics and flight recorder.
  - `layers/triton_lookup.py`: Triton GPU kernel and PyTorch fallback for parallel codebook snapping.
  - `tokenizers/code_tmot.py`: AST-based Python code structural parser.
  - `tokenizers/text_tmot.py`: Akshara-syllable tokenizer for Indic languages.
  - `utils/checker.py`: Layout verification safety decorator.
  - `utils/dashboard.py`: Live terminal inference trace inspector.
  - `utils/data_ingestion.py`: Streaming dataset loader.
- `train.py`: Multi-modal training harness (Text, Code, Sensor data).
- `train_indic.py`: Fine-tuning script for Indic languages.
- `configs/text_small.yaml`: Hyperparameter profile.
- `tests/`: Complete unit testing suite for AST isolation, numerical parity, and gradient dynamics.
- `paper/`: Complete academic manuscript LaTeX template and BibTeX bibliography.

## Running Tests

```bash
python -m unittest discover tests
```

## Training

```bash
python train.py
python train_indic.py
```
