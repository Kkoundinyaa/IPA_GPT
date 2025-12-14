import os
import pathlib
from typing import List
from datasets import load_dataset, concatenate_datasets, ClassLabel
import evaluate
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from hf_wrapper import GPTForSequenceClassification
from model import GPT, GPTConfig
from tokenizer import load_tokenizer, eod_token
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Config ---
train_lang = 'both'      
eval_lang = 'both'    

BASE_DIR = pathlib.Path("/fs/scratch/PAS2836/ipa_gpt")
TOKENIZER_DIR = pathlib.Path("/fs/ess/PAS2836/ipa_gpt/tokenizers")

CHECKPOINTS = {
    "ipa": BASE_DIR / "checkpoints/russian_polish_ipa_12_5_50k/ckpt.pt",
    "normal": BASE_DIR / "checkpoints/russian_polish_normal_12_5_50k/ckpt.pt",
}

TOKENIZERS = {
    "ipa": (
        TOKENIZER_DIR / "bpe-rus-pol-ipa-number-preservation-vocab.json",
        TOKENIZER_DIR / "bpe-rus-pol-ipa-number-preservation-merges.txt",
    ),
    "normal": (
        TOKENIZER_DIR / "bpe-rus-pol-normal-number-preservation-vocab.json",
        TOKENIZER_DIR / "bpe-rus-pol-normal-number-preservation-merges.txt",
    ),
}

LANG_TO_DATASET = {
    "ru": "iggy12345/xnli-ru-ipa",
    "pl": "iggy12345/cdsc-e-ipa"
}

args = {
    'epochs': 8,
    'context_size': 1024,
    'learning_rate': 2e-5,
    'batch_size': 16,
    'hf_cache_dir': pathlib.Path('cache'),
    'device': 'cuda',
}

def load_pretrained_model(path: pathlib.Path, device: str = 'cuda') -> GPT:
    checkpoint = torch.load(path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    filtered = {k: v for k, v in state_dict.items()
                if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict({**model.state_dict(), **filtered})
    return model.to(device)

def flatten_multi_features(examples, features: List[str]) -> List[str]:
    sep = f'\n\n{eod_token}\n\n'
    return [sep.join([x or '' for x in items]) for items in zip(*[examples[f] for f in features])]

def get_fields(example, model_type):
    if model_type == "ipa":
        if 'premise-phoneme' in example:
            return ['premise-phoneme', 'hypothesis-phoneme']
        elif 'sentence_A-phoneme' in example:
            return ['sentence_A-phoneme', 'sentence_B-phoneme']
    if 'premise' in example:
        return ['premise', 'hypothesis']
    return ['sentence_A', 'sentence_B']

def load_and_preprocess(dataset_name, split, tokenizer, model_type):
    ds = load_dataset(dataset_name, split=split, cache_dir=str(args['hf_cache_dir']))
    if 'label' in ds.features and not isinstance(ds.features['label'], ClassLabel):
        ds = ds.cast_column("label", ClassLabel(names=['entailment', 'neutral', 'contradiction']))
    sample = ds[0]
    fields = get_fields(sample, model_type)

    def preprocess(examples):
        features = flatten_multi_features(examples, fields)
        return tokenizer(features, truncation=True, max_length=args['context_size'])

    return ds.map(preprocess, batched=True)

metric = evaluate.load("xnli", "en")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.from_numpy(logits).argmax(dim=-1)
    labels = torch.from_numpy(labels)

    correct = (preds == labels).sum().item()
    total = len(labels)
    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0)
    }

# === Run both IPA and NORMAL models ===
for model_type in ['ipa', 'normal']:
    print(f"\n🔧 Running setup for {model_type.upper()} model")
    vocab_path, merges_path = TOKENIZERS[model_type]
    tokenizer = load_tokenizer(vocab_path, merges_path)
    base_model = load_pretrained_model(CHECKPOINTS[model_type], args['device'])
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.padding_side = tokenizer.padding_side
    model = GPTForSequenceClassification(base_model, num_classes=3).to(args['device'])

    if train_lang == 'both':
        train_ru = load_and_preprocess(LANG_TO_DATASET['ru'], 'train', tokenizer, model_type)
        train_pl = load_and_preprocess(LANG_TO_DATASET['pl'], 'train', tokenizer, model_type)
        train_dataset = concatenate_datasets([train_ru, train_pl]).shuffle(seed=42)
    else:
        train_dataset = load_and_preprocess(LANG_TO_DATASET[train_lang], 'train', tokenizer, model_type)

    if eval_lang == 'both':
        eval_ru = load_and_preprocess(LANG_TO_DATASET['ru'], 'validation', tokenizer, model_type)
        eval_pl = load_and_preprocess(LANG_TO_DATASET['pl'], 'train', tokenizer, model_type)       #change to pl[20%] when required
        eval_dataset = concatenate_datasets([eval_ru, eval_pl])
    elif eval_lang == 'ru':
        eval_dataset = load_and_preprocess(LANG_TO_DATASET['ru'], 'validation', tokenizer, model_type)
    else:
        eval_dataset = load_and_preprocess(LANG_TO_DATASET['pl'], 'train', tokenizer, model_type)   #change to pl[20%] when required

    output_dir = pathlib.Path(f"./training_outputs_rupl/{train_lang}2{eval_lang}_{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        metric_for_best_model="precision",
        load_best_model_at_end=True,
        learning_rate=args['learning_rate'],
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        num_train_epochs=args['epochs'],
        weight_decay=0.01,
        logging_steps=1000,
        logging_dir='./logs',
        fp16=True,
        disable_tqdm=False,
        warmup_ratio=0.3,
        seed=42,
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"Training {model_type.upper()} model on {train_lang.upper()} → Evaluating on {eval_lang.upper()}")
    trainer.train()

    print(f"Final evaluation on {eval_lang.upper()} for model {model_type.upper()}")
    results = trainer.evaluate()
    print(results)
