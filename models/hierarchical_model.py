import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from models.custom_attention import PropagandaAttention

class HierarchicalAttentionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        
        # Instead of directly replacing the attention mechanism,
        # we'll modify specific parameters or use the output in a different way
        
        # Sentence-level attention
        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        # Custom propaganda attention for final representation
        self.propaganda_attention = PropagandaAttention(self.bert.config.hidden_size)
        
        # Domain adaptation layer
        self.domain_adapter = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.bert.config.hidden_size))
        
        # Contrastive projection head
        self.contrastive_projection = nn.Linear(self.bert.config.hidden_size, 128)
        
        # Classifier
        self.classifier = nn.Linear(self.bert.config.hidden_size, 10)

    def forward(self, input_ids, attention_mask):
        # Word-level features from DistilBert (unmodified)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        word_level = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
        
        # Calculate mean over sequence dimension to get sentence representation
        sentence_repr = word_level.mean(dim=1)  # Shape: [batch_size, hidden_dim]
        
        # MultiheadAttention expects inputs of shape [seq_len, batch_size, hidden_dim]
        # Reshape accordingly
        query = sentence_repr.unsqueeze(0)  # Shape: [1, batch_size, hidden_dim]
        key_value = word_level.transpose(0, 1)  # Shape: [seq_len, batch_size, hidden_dim]
        
        # Apply sentence-level attention
        sentence_level, attn_weights = self.sentence_attention(
            query, key_value, key_value)  # Shape: [1, batch_size, hidden_dim]
        
        # Remove the sequence dimension
        sentence_level = sentence_level.squeeze(0)  # Shape: [batch_size, hidden_dim]
        
        # Apply custom propaganda attention to further process the sentence representation
        # Create a attention mask (all 1s since we're not masking anything at this stage)
        custom_mask = torch.ones(sentence_level.size(0), 1, device=sentence_level.device)
        propaganda_level, _ = self.propaganda_attention(sentence_level.unsqueeze(1), attention_mask=custom_mask)
        propaganda_level = propaganda_level.squeeze(1)  # Shape: [batch_size, hidden_dim]
        
        # Domain adaptation
        adapted = self.domain_adapter(propaganda_level)
        
        # Contrastive projection
        contrastive_emb = self.contrastive_projection(adapted)
        
        # Classification
        logits = self.classifier(adapted)
        logits = torch.sigmoid(logits)

        return {
                'logits': logits,
                'contrastive_emb': contrastive_emb,
                'word_attentions': attn_weights
            }