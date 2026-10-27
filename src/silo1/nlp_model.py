"""
KNOWN ISSUE — minor-project prototype model, do not use (docs/AUDIT.md §4B).
XLMRobertaModel(config) builds a RANDOMLY INITIALISED encoder (no from_pretrained) and then freezes it, so the heads
train on random features. In the prototype run it was never even used: the training script crashed on a missing
`is_functional` attribute and silently fell back to a toy model. Silo 1 now uses frozen pretrained sentence
embeddings + MLPHead (src/fl/models.py, experiments/silo1_embed_cache.py).
"""

import logging
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig

logger = logging.getLogger("urbanfl.silo1.nlp")

class CrisisClassifier(nn.Module):
    """
    Industry-scale XLM-RoBERTa multi-task crisis classifier.
    Optimized for low VRAM using a frozen shared encoder base, with separate
    heads for categorical threat classification and regression-based severity.
    """
    def __init__(self, num_classes: int = 5, freeze_encoder: bool = True):
        super().__init__()
        # Use a lightweight config for memory efficiency on edge nodes unless weights are specified
        config = XLMRobertaConfig(
            vocab_size=250002, hidden_size=768, num_hidden_layers=6, 
            num_attention_heads=12, intermediate_size=3072
        )
        self.encoder = XLMRobertaModel(config)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Multi-task heads
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, num_classes)
        )
        
        self.severity_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid() # Bound severity between 0 and 1
        )
        logger.info(f"CrisisClassifier initialized. Frozen encoder: {freeze_encoder}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Base encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # (batch_size, hidden_size)
        
        # Classification (CrossEntropy target)
        category_logits = self.classification_head(pooled_output)
        
        # Severity ranking (MSE target)
        severity_score = self.severity_head(pooled_output)
        
        return {
            "category_logits": category_logits,
            "severity_score": severity_score.squeeze(-1)
        }
