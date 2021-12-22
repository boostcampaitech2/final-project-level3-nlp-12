import torch
import torch.nn as nn
import torch.nn.functional as F

from attrdict import AttrDict
from utils import read_json
from base import BaseModel
from transformers import AutoModel, AutoModelForSequenceClassification
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
        self.model.electra.encoder.layer[3] = BeepKcElectraResMLayer(self.model, 3)
        self.model.electra.encoder.layer[7] = BeepKcElectraResMLayer(self.model, 7)

        # deletion top layer
        del self.model.electra.encoder.layer[11]
        # del self.model.electra.encoder.layer[10]

    def forward(self, inputs):
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
        params.mem_size = 768 * 768
        # params.mem_size = 128 * 128
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


class KcElectraDnnV1Model(BaseModel):
    def __init__(
        self,
        name="beomi/beep-KcELECTRA-base-hate",
        hidden_size=768,
        dropout_rate=0.5,
        num_classes=3
    ):
        super().__init__()
        
        self.embeddings = AutoModel.from_pretrained(name)
        
        for _ in range(8):
            del self.embeddings.encoder.layer[-1]
            
        self.head = KcElectraDnnV1Head(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
            
    def forward(self, inputs):
        x = self.embeddings(**inputs)
        x = self.head(x.last_hidden_state)
        
        return x
        
        
class KcElectraDnnV1Head(nn.Module):
    '''
    A class as a head of PLM(KcELECTRA) for convolutional and sequential process (concat + bilstm + lstm)
    
    Args:
        hidden_size (int): a value from PLM hidden size
        dropout_rate (float): dropout rate
        num_classes (int): number of classes
    '''
    def __init__(self, hidden_size, dropout_rate, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=5, padding=2)
        
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.biLSTM = nn.LSTM(
            input_size=hidden_size, # output from PLM
            hidden_size=int(hidden_size/2),
            num_layers=2,
            batch_first=True,
            bidirectional=True # output dim x 2
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size, # output from PLM
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, embeddings):
        x = embeddings.transpose(1, 2).contiguous() # output => (B, D, L)
        
        conv1 = nn.functional.relu(self.conv1(x)) # output => (B, out_channels(D/3), L)
        conv2 = nn.functional.relu(self.conv2(x))
        conv3 = nn.functional.relu(self.conv3(x))
        
        concat = torch.cat((conv1, conv2, conv3), -2).transpose(1, 2).contiguous() # output => (B, D, L) => (B, L, D)
        conv_output = self.dropout1(concat) # output => (B, L, D)
        
        # residual mapping
        bi_lstm_output, (_, _) = self.biLSTM(conv_output + embeddings) # output => (B, L, D) 
        bi_lstm_output = nn.functional.gelu(bi_lstm_output)
        lstm_output, (_, _) = self.lstm(bi_lstm_output) # output => (B, L, D)
        lstm_output = nn.functional.gelu(lstm_output)
        lstm_output = self.dropout2(lstm_output)
        
        output = F.max_pool1d(lstm_output, kernel_size=lstm_output.shape[2]).squeeze(-1) # output => (B, L, 1) => (B, L)        
        
        output = self.fc(output) # output => (B, 3)
        
        return output
    
    
class KcElectraDnnV2Model(BaseModel):
    def __init__(
        self,
        name="beomi/beep-KcELECTRA-base-hate",
        hidden_size=768,
        dropout_rate=0.5,
        num_classes=3
    ):
        super().__init__()
        
        self.hate_embeddings = AutoModel.from_pretrained(name)
        self.bias_embeddings = AutoModel.from_pretrained('beomi/beep-KcELECTRA-base-bias')
        
        for _ in range(8):
            del self.hate_embeddings.encoder.layer[-1]
        for _ in range(8):
            del self.bias_embeddings.encoder.layer[-1]
            
        self.head = KcElectraDnnV2Head(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
            
    def forward(self, inputs):
        hate_embeds = self.hate_embeddings(**inputs)
        bias_embeds = self.bias_embeddings(**inputs)
        
        x = self.head(hate_embeds.last_hidden_state, bias_embeds.last_hidden_state)
        
        return x
        
        
class KcElectraDnnV2Head(nn.Module):
    '''
    A class as a head of PLM(KcELECTRA) for convolutional and sequential process (concat + bilstm + lstm)
    
    Args:
        hidden_size (int): a value from PLM hidden size
        dropout_rate (float): dropout rate
        num_classes (int): number of classes
    '''
    def __init__(self, hidden_size, dropout_rate, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=5, padding=2)
        
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.biLSTM = nn.LSTM(
            input_size=(hidden_size*2), # output from PLM
            hidden_size=int(hidden_size/2),
            num_layers=2,
            batch_first=True,
            bidirectional=True # output dim x 2
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size, # output from PLM
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, hate_embeds, bias_embeds):
        emb_h = hate_embeds.transpose(1, 2).contiguous() # output => (B, D, L)
        emb_b = bias_embeds.transpose(1, 2).contiguous() # output => (B, D, L)
        
        conv1_h = nn.functional.relu(self.conv1(emb_h)) # output => (B, out_channels(D/3), L)
        conv2_h = nn.functional.relu(self.conv2(emb_h))
        conv3_h = nn.functional.relu(self.conv3(emb_h))
        
        conv1_b = nn.functional.relu(self.conv1(emb_b)) # output => (B, out_channels(D/3), L)
        conv2_b = nn.functional.relu(self.conv2(emb_b))
        conv3_b = nn.functional.relu(self.conv3(emb_b))
        
        # output => (B, D x 2, L) => (B, L, D x 2)
        concat = torch.cat((conv1_h, conv2_h, conv3_h, conv1_b, conv2_b, conv3_b), dim=1).transpose(1, 2).contiguous() 
        conv_output = self.dropout1(concat) # output => (B, L, D x 2)
        
        # residual mapping
        bi_lstm_output, (_, _) = self.biLSTM(conv_output + torch.cat((hate_embeds, bias_embeds), dim=2)) # output => (B, L, D)
        bi_lstm_output = nn.functional.gelu(bi_lstm_output)
        lstm_output, (_, _) = self.lstm(bi_lstm_output) # output => (B, L, D)
        lstm_output = nn.functional.gelu(lstm_output)
        lstm_output = self.dropout2(lstm_output)
        
        output = F.max_pool1d(lstm_output, kernel_size=lstm_output.shape[2]).squeeze(-1) # output => (B, L, 1) => (B, L)        
        
        output = self.fc(output) # output => (B, 3)
        
        return output