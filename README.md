# IPA_GPT — Phonetic (IPA) Tokenization for Multilingual LLM Fine-Tuning

**TL;DR:** This project investigates whether **IPA-based (phonetic) tokenization** can improve cross-lingual transfer and robustness in GPT/Transformer-based models, especially for low-resource and cross-script language settings.

---

## Motivation
Standard subword tokenizers often fail to capture phonetic similarity across languages, particularly for low-resource languages and cross-script transfer (e.g., Cyrillic ↔ Latin).  
This repository explores an alternative pipeline where text is converted to **IPA (International Phonetic Alphabet)** before tokenization and fine-tuning, with the goal of improving generalization across languages and datasets.

---

## Repository Structure

```
IPA_GPT/
├── notebooks/
│   ├── dataset-experiments/   # Dataset-specific IPA vs baseline experiments
│   └── debugging/             # Sanity checks, tokenizer/debug notebooks
├── scripts/
│   ├── finetuning/            # Main fine-tuning scripts
│   ├── model/                 # Model definitions / extensions
│   └── wrappers/              # Training / evaluation wrappers
├── results/                   # Tables and summaries of experimental results
├── assets/                    # Figures used in documentation
├── .gitignore
└── README.md
```

---

## Experiments Covered
The `notebooks/dataset-experiments/` folder contains experiments across multiple NLP benchmarks and language pairs, including:

- **GLUE tasks:** MRPC, RTE, SST-2, WNLI
- **Cross-lingual tasks:** XNLI
- **Language pairs:** English, Spanish → English, Russian ↔ Polish
- **Settings:** Baseline subword tokenization vs IPA-based tokenization

Each notebook focuses on **one dataset or language pair**, enabling controlled comparison between tokenization strategies.

---

## Method Overview
1. Text → IPA conversion using phonemization rules  
2. Tokenizer construction (baseline vs IPA-based)  
3. Model fine-tuning on downstream classification tasks  
4. Evaluation using task-specific metrics (accuracy / F1 where applicable)  
5. Qualitative analysis of failure cases and transfer behavior  

---

## Results Summary
A detailed summary of results is maintained in the `results/` directory.

**High-level observations:**
- IPA tokenization shows promising improvements in some cross-lingual and low-resource settings
- Gains are task-dependent and sensitive to phonemization quality
- Certain datasets exhibit degradation due to information loss during phonetic conversion

(Results tables and figures will be progressively added as experiments are finalized.)

---

## Reproducibility
- Frameworks: PyTorch, Hugging Face Transformers
- Environment: Jupyter, Google Colab, and HPC (OSC cluster)
- Experiments are notebook-driven for transparency and analysis

---

## Limitations & Notes
- IPA conversion quality depends heavily on language-specific phonemizers
- Some scripts may lose orthographic or semantic cues during transliteration
- This repository represents exploratory research and experimentation

---

## Future Work
- Consolidating notebook logic into unified training scripts
- Scaling experiments to larger multilingual corpora
- Evaluation on generative and zero-shot tasks
- Extending experiments to pretraining-level settings

---

## Credits
This project builds on open-source NLP tooling including PyTorch, Hugging Face Transformers, and phonemization libraries.
