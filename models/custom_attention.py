import torch
import torch.nn.functional as F
from torch import nn

class PropagandaAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # In DistilBert, hidden_states is used for query, key, and value
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        
        # Apply mask (if provided)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        # Compute attention probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        # Compute final output
        output = torch.matmul(attn_probs, value)
        
        # Always return a tuple with the output and attention probabilities
        # This matches what DistilBert expects
        return (output, attn_probs)