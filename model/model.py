import torch.nn as nn
import torch.nn.functional as F

from attrdict import AttrDict
from utils import read_json
from base import BaseModel
from transformers import AutoModelForSequenceClassification
from transformers.modeling_utils import apply_chunking_to_forward
from utils.memory import HashingMemoryProduct

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class BeepKcElectraHateModel(BaseModel):
    def __init__(self, name="beomi/beep-KcELECTRA-base-hate", num_classes=3):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_classes)
        
    def forward(self, inputs):
        return self.model(**inputs)

class BeepKcElectraResMModel(BaseModel):
    def __init__(self, name="beomi/beep-KcELECTRA-base-hate", num_classes=3):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_classes)

        # self.embeddings = self.model.electra.embeddings
        self.model.electra.encoder.layer[3] = BeepKcElectraResMLayer(self.model, 3)
        self.model.electra.encoder.layer[7] = BeepKcElectraResMLayer(self.model, 7)
        # self.encoders = self.model.electra.encoder
        # self.classifier = self.model.classifier

    def forward(self, inputs):
        # embedding_output = self.embeddings(**inputs)
        # encoder_outputs = self.encoders(embedding_output)
        # return self.classifier(encoder_outputs)
        return self.model(**inputs)

class BeepKcElectraResMLayer(nn.Module):
    def __init__(self, model, layer_num):
        super().__init__()
        self.attention = model.electra.encoder.layer[layer_num].attention
        self.intermediate = model.electra.encoder.layer[layer_num].intermediate
        self.output = ResM(model.electra.encoder.layer[layer_num].output)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, 0, 1, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) # ffn + pkm
        return layer_output

class ResM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dense = model.dense
        self.dropout = model.dropout
        self.LayerNorm = model.LayerNorm

        params = read_json('./pkm_config.json')
        params = AttrDict(params)
        params.mem_size = 128 * 128
        params.n_indices = params.mem_size
        params.mem_product_quantization = True
        params.mem_sparse = False
        self.pkm = HashingMemoryProduct(3072, 768, params) # input_dim, output_dim, params

    def forward(self, hidden_states, input_tensor):
        dense_hidden_states = self.dense(hidden_states)
        dense_hidden_states = self.dropout(dense_hidden_states)
        pkm_hidden_states = self.pkm(hidden_states)
        hidden_states = self.LayerNorm(dense_hidden_states + pkm_hidden_states + input_tensor)
        return hidden_states
