

# world-model-kernels

High-performance kernels for **world models** and **model-based planning**.

This library focuses on the core numeric workloads that appear in agents like
Dreamer / PlaNet / MuZero:

- Rolling out a learned dynamics model over many imagined steps and branches
- Computing discounted returns / value backups

The goal is to expose these as clean, testable Python functions while giving
you room to write and optimize custom kernels (Triton / CUDA / CPU).

---

## Features

- ✅ **Discounted return kernel**
  - Triton implementation for GPU
  - CPU fallback (vectorized PyTorch)
- 🧪 **Tests** comparing kernels to a simple reference implementation
- 🔧 **Pluggable backends**
  - `backend="cpu"` – always use CPU implementation
  - `backend="triton"` – use Triton if available, otherwise fall back

Planned:

- 🚧 Fused **rollout kernels** (multi-step latent dynamics + reward)
- 🚧 Kernels for tree backups (MuZero-style planners)

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/world-model-kernels.git
cd world-model-kernels

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
