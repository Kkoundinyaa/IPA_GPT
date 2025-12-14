import torch
from model import GPT, GPTConfig
from gpt_classifier import GPTForSequenceClassification  # move your classifier code here
from tokenizer import load_tokenizer
from transformers import GPT2Tokenizer

def load_pretrained_model(path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    pretrained_model = GPT(gptconf)
    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model_dict = pretrained_model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_state_dict)
    pretrained_model.load_state_dict(model_dict)
    pretrained_model.to(device)

    return pretrained_model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model_path = "/fs/scratch/PAS2836/ipa_gpt/checkpoints/openwebtext_ipa_multi_node_12_5_medium_50k/ckpt.pt"
    base_model = load_pretrained_model(model_path, device)

    # Manually add pad_token_id and padding_side to config
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.padding_side = tokenizer.padding_side

    model = GPTForSequenceClassification(base_model).to(device)
    torch.save(model.state_dict(), "finetune_ready_model.pt")
