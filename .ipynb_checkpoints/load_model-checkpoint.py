import torch
from model import GPT, GPTConfig

def load_pretrained_model(path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    pretrained_model = GPT(gptconf)
    state_dict = checkpoint['model']

    # Remove '_orig_mod.' prefix from keys
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Load only matching weights
    model_dict = pretrained_model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_state_dict)
    pretrained_model.load_state_dict(model_dict)

    pretrained_model.to(device)
    return pretrained_model
