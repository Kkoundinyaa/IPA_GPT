import torch
import torch.nn as nn

class GPTForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_labels=2, problem_type="classification"):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.hidden_size = pretrained_model.config.n_embd

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, 1 if problem_type == "regression" else num_labels)

        self.pad_token_id = getattr(pretrained_model.config, "pad_token_id", 0)
        self.padding_side = "right"

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.pretrained_model(input_ids)  # (batch, seq_len, hidden_size)
        pooled_output = hidden_states[:, 0, :]  # Use first token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.problem_type == "classification":
                if torch.any((labels < 0) | (labels >= self.num_labels)):
                    print("❌ Invalid label found:", labels)
                    print(f"⚠ Expected label range: [0, {self.num_labels - 1}]")
                    exit()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "regression":
                logits = torch.clamp(logits.view(-1), 0, 5)  # ⬅️ For STS-B etc.
                labels = labels.view(-1).float()
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits
