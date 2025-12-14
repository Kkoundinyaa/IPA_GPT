# IPA_GPT — Phonetic (IPA) Tokenization for Multilingual LLM Fine-Tuning

**TL;DR:** This repository explores **IPA-based (phonetic) tokenization** as an alternative to standard subword tokenization for multilingual and cross-lingual fine-tuning of GPT-style models, with a focus on low-resource and cross-script language settings.

---

## Motivation
Standard subword tokenizers often fragment words inconsistently across scripts and languages, limiting cross-lingual transfer.  
This project investigates whether converting text into **IPA (International Phonetic Alphabet)** before tokenization can help models better capture phonetic similarity across languages, improving robustness and generalization.

---

## Repository Structure

```
IPA_GPT/
├── notebooks/
│   ├── dataset-experiments/   # Dataset- and language-pair–specific experiments
│   └── debugging/             # Tokenizer, dataloader, and sanity-check notebooks
├── scripts/
│   ├── finetuning/            # Main fine-tuning entry points
│   ├── model/                 # Model definitions and extensions
│   └── wrappers/              # Training / evaluation wrappers
├── results/                   # Tables and summaries of experimental results
├── assets/                    # Figures used in documentation
├── .gitignore
└── README.md
```

---

## Experiments Covered
Experiments are organized primarily in `notebooks/dataset-experiments/` and include:

- **GLUE tasks:** MRPC, RTE, SST-2, WNLI  
- **Cross-lingual tasks:** XNLI  
- **Language pairs:** English, Spanish → English, Russian ↔ Polish  
- **Comparisons:** Baseline subword tokenization vs IPA-based tokenization  

Each notebook focuses on a **single dataset or language-pair setting**, enabling controlled and interpretable comparisons.

---

## Fine-Tuning Pipeline (High Level)

The fine-tuning workflow follows these steps:

1. **Phonemization:** Convert raw text to IPA using language-specific phonemization rules  
2. **Tokenization:** Apply either a baseline subword tokenizer or an IPA-based tokenizer  
3. **Model Setup:** Load a GPT-style model compatible with the NanoGPT architecture  
4. **Fine-Tuning:** Train on downstream classification tasks  
5. **Evaluation:** Measure performance using task-appropriate metrics (accuracy / F1)  
6. **Analysis:** Inspect failure cases and cross-lingual transfer behavior  

Fine-tuning currently supports **classification tasks**. Seq2seq tasks are not yet supported.

---

## Configuration System
Experiments are configured using **cascading JSON configuration files**:

- A base configuration defines directory paths and defaults
- Task- or language-specific configs override model, tokenizer, dataset, and hyperparameters
- Multiple config files can be passed together, with later files overriding earlier ones

This design enables reproducible experiments and easy comparison across settings.

---

## Language & Dataset Registry
A centralized **language database** defines:
- Languages
- Associated datasets
- Task types (classification)
- Input features, labels, and splits
- Number of classes per task

This allows the same training pipeline to generalize across multiple datasets and languages with minimal code changes.

---

## Results Summary
Results and comparisons are summarized in the `results/` directory.

**Observed trends so far:**
- IPA-based tokenization shows promise in certain cross-lingual and low-resource settings
- Improvements are task-dependent and sensitive to phonemization quality
- Some datasets experience degradation due to loss of orthographic or semantic cues

Detailed tables and plots will be added as experiments are finalized.

---

## Reproducibility
- Frameworks: PyTorch, Hugging Face Transformers
- Execution: Jupyter notebooks, local runs, and SLURM-based HPC environments
- Experiments are notebook-driven for transparency and analysis

---

## Limitations
- IPA conversion quality varies across languages
- Phonemization may remove useful orthographic information
- Results should be interpreted as exploratory research findings

---

## Future Work
- Consolidation of notebook logic into a unified training pipeline
- Scaling experiments to larger multilingual corpora
- Evaluation on zero-shot and generative tasks
- Extension to pretraining-level experiments

---

## Credits
This project builds on open-source tooling including PyTorch, Hugging Face Transformers, NanoGPT-style model implementations, and phonemization libraries.
