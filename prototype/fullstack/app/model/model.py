import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification
from base import BaseModel


# class MyEfficientNet(nn.Module) :
#     '''
#     EfiicientNet-b4의 출력층만 변경합니다.
#     한번에 18개의 Class를 예측하는 형태의 Model입니다.
#     '''
#     def __init__(self, num_classes: int = 18) :
#         super(MyEfficientNet, self).__init__()
#         self.EFF = EfficientNet.from_pretrained('efficientnet-b4', in_channels=3, num_classes=num_classes)
    
#     def forward(self, x) -> torch.Tensor:
#         x = self.EFF(x)
#         x = F.softmax(x, dim=1)
#         return x
    
class BeomiModel(BaseModel):
    def __init__(self, name="beomi/beep-KcELECTRA-base-hate", num_classes=3):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_classes)
        
    def forward(self, inputs):
        return self.model(**inputs)