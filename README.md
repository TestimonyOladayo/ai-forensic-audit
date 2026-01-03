# ai-forensic-audit
Mechanistic Interpretability tools for auditing safety circuits in Transformer models
# 🔍 AI Forensic Audit: Layer 7 Safety Circuit Map

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Library: TransformerLens](https://img.shields.io/badge/Library-TransformerLens-orange.svg)](https://github.com/transformer-lens/TransformerLens)
[![Architecture: GPT-2](https://img.shields.io/badge/Architecture-GPT--2-green.svg)](https://huggingface.co/gpt2)

##  Executive Summary
This project provides a **Mechanistic Interpretability** framework for auditing the internal reasoning of GPT-2. 
![AI Forensic Audit Map](newplot%20(1).png)

As we shift from simple LLMs to **Autonomous Agents**, "Black Box" transparency is no longer optional. This audit maps the **Activation Delta ($\Delta$)** of specific neurons to identify "Safety Triggers"—internal circuits that respond to hostile intent before an output is even generated.
---
##  Strategic Design Decisions

### 1. Resource Constraint Engineering (4GB RAM)
To enable high-level research on edge hardware (Google Colab Free Tier), I implemented:
* **Gradient-Free Caching:** Using `run_with_cache` to extract internal states without storing unnecessary backprop data.
* **CPU-Centric Inference:** Optimized for environments where VRAM is unavailable or highly restricted.

# 2. Binary Compatibility Fix
Resolved a critical **C-header mismatch** between NumPy 2.0 and TransformerLens. 
* **The Solution:** Forced a stable environment pin (`numpy==1.26.4`) and a clean kernel restart to flush binary caches.

---

## Setup & Execution

### Prerequisites
* Python 3.9+
* A terminal or Google Colab session

### Installation
```bash
# Force the correct binary environment
pip uninstall -y numpy transformer_lens
pip install numpy==1.26.4 transformer_lens plotly --quiet
pip uninstall -y numpy transformer_lens
pip install numpy==1.26.4 transformer_lens plotly --quiet
