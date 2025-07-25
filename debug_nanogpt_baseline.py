#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset, load_from_disk
import evaluate
import torch
from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from model import GPT, GPTConfig
from tokenizer import load_tokenizer
from transformers import GPT2Tokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np


# In[2]:


# 2. Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token


# In[3]:


class GPTForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes=2):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        self.hidden_size = pretrained_model.config.n_embd
        self.pad_token_id = pretrained_model.config.pad_token_id

        self.classifier = nn.Linear(self.hidden_size, num_classes, bias=False)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.pretrained_model(input_ids)  # (batch_size, seq_len, hidden_size)
        logits = self.classifier(hidden_states)
        # print(input_ids)
        # input()

        batch_size, sequence_length = input_ids.shape[:2]

        # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
        non_pad_mask = (input_ids != self.pad_token_id).to(logits.device, torch.int32)
        token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pooled_logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=None,
            attentions=None,
        )


# In[7]:


def load_pretrained_model(path, device='cuda'):
    # Load the pretrained model
    print(f"Loading pretrained model from {path}")
    checkpoint = torch.load(path, map_location=device)

    # Create the nanoGPT instance to load in saved weights
    gptconf = GPTConfig(**checkpoint['model_args'])
    pretrained_model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Clean up the saved state
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Only load the parameters that match the checkpoint weights
    model_dict = pretrained_model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_state_dict)
    pretrained_model.load_state_dict(model_dict)
    pretrained_model.to(device)

    return pretrained_model


# In[8]:


model = "/fs/scratch/PAS2836/ipa_gpt/checkpoints/openwebtext_ipa_multi_node_12_5_medium_50k/ckpt.pt"
device = "cuda"


# In[9]:


tokenizer


# In[10]:


base_model = load_pretrained_model(model, device)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.padding_side = tokenizer.padding_side
model = GPTForSequenceClassification(base_model).to(device)


# In[8]:


total_trainable_params = sum(param.numel() for param in model.parameters())
print(f"Total number of trainable parameters: {total_trainable_params}")


# In[9]:


# 1. Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]


# In[10]:


# 3. Preprocessing
def preprocess(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding=True,
        max_length=128,
    )

encoded_train = train_dataset.map(preprocess)
encoded_eval = eval_dataset.map(preprocess)

# 4. Metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# 5. Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-sst2-cls",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
    report_to="none",  # No wandb
    warmup_ratio=0.1,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Train!
trainer.train()


# In[ ]:




