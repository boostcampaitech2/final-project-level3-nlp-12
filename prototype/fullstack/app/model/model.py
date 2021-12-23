import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification
from base import BaseModel


class BeomiModel(BaseModel):
    def __init__(self, name="beomi/beep-KcELECTRA-base-hate", num_classes=3):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_classes)
        
    def forward(self, inputs):
        return self.model(**inputs)

