# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install from git (standard installation)
pip install git+https://github.com/lamm-mit/PRefLexOR.git

# Install for development (editable installation)
git clone https://github.com/lamm-mit/PRefLexOR.git
cd PRefLexOR
pip install -r requirements.txt
pip install -e .

# Optional: Install Flash Attention for performance
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Running Applications
```bash
# Launch business application suite (Financial Risk, Medical Diagnosis, etc.)
./launch_apps.sh

# Run individual Streamlit apps
cd financialrisk && streamlit run app.py --server.port 8501
cd medical_diagnosis && streamlit run app.py --server.port 8502
# ... etc for other apps
```

### Testing Commands
```bash
# Test Ollama connection (required for business apps)
python financialrisk/test_ollama.py

# No formal test suite exists - consider adding pytest framework
```

## Architecture Overview

**PRefLexOR** is a framework for preference-based recursive language modeling that combines:
- **ORPO (Odds Ratio Preference Optimization)** - Phase I training
- **DPO (Direct Preference Optimization)** with EXO (Efficient Exact Optimization) - Phase II training
- **Thinking Tokens** - Special tokens `<|thinking|>` and `<|/thinking|>` for explicit reasoning
- **Active Learning** - Dynamic dataset generation during training
- **Recursive Reasoning** - Iterative self-improvement through reflection

### Core Package Structure
```
PRefLexOR/
├── active_trainer.py    # ORPO/DPO trainers with dynamic dataset generation
├── inference.py         # Recursive reasoning with thinking tokens
└── utils.py            # RAG utilities, OpenAI integration, token extraction
```

### Business Applications
Separate Streamlit applications demonstrate PRefLexOR in various domains:
- `financialrisk/` - Financial risk assessment
- `medical_diagnosis/` - Medical diagnosis support
- `supply_chain/` - Supply chain risk management
- `legal_analysis/` - Legal document analysis
- `investment_research/` - Investment research
- `product_development/` - Product development strategy

Each app requires Ollama running locally with `llama3.1:8b-instruct-q4_K_M` model.

### Training Workflow

1. **Phase I - ORPO Training**: Uses `PRefLexORORPOTrainer` for structured thought integration
2. **Phase II - DPO Training**: Uses `PRefLexORDPOTrainer` for independent reasoning development
3. **Recursive Inference**: `recursive_response_from_thinking()` implements iterative refinement

### Key Classes and Functions

#### `PRefLexOR/active_trainer.py`
- `PRefLexORORPOTrainer`: Extends TRL's ORPOTrainer with dynamic dataset generation
- `PRefLexORDPOTrainer`: Extends TRL's DPOTrainer with thinking token support
- Both trainers support on-the-fly dataset generation and RAG integration

#### `PRefLexOR/inference.py`
- `recursive_response_from_thinking()`: Core algorithm for recursive reasoning
- `generate_local_model()`: Local model inference with thinking token support
- Supports model-critic feedback loops

#### `PRefLexOR/utils.py`
- `generate_GPT_MistralRS()`: Dataset generation using OpenAI API
- `extract_text_*()`: Functions to extract thinking/reflection sections
- RAG utilities for context retrieval

### Special Tokens
- `<|thinking|>...<|/thinking|>` - Explicit reasoning sections
- `<|reflect|>...<|/reflect|>` - Reflection sections for recursive improvement

### Model Integration
- HuggingFace Transformers base
- LoRA/PEFT support for efficient fine-tuning
- Pre-trained models: `lamm-mit/PRefLexOR_*` on HuggingFace Hub
- Flash Attention 2 support for performance

### Key Dependencies
- `torch` - PyTorch framework
- `transformers>=4.45` - HuggingFace transformers
- `trl` - Transformer Reinforcement Learning
- `peft` - Parameter Efficient Fine-Tuning
- `llama-index-*` - RAG and vector indexing
- `openai` - Dataset generation
- `accelerate`, `bitsandbytes` - Training optimization