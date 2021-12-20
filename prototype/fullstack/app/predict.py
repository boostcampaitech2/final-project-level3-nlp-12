from transformers import AutoTokenizer, pipeline
import torch
from tqdm import tqdm
from load_data import NhDataloader
import joblib


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nh_model = torch.load('/opt/ml/final-project-level3-nlp-12/prototype/fullstack/app/model/nh_model.pt', map_location=device) ## 경로 맞게!

    with open('/opt/ml/final-project-level3-nlp-12/prototype/fullstack/app/model/nh_classifier.pkl', 'rb') as f: ## 경로 맞게!
        nh_classifier = joblib.load(f)

    nh_model.to(device)
    nh_model.eval()
    
    return nh_model, nh_classifier, 

def load_dataloader(data):
    nh_tokenizer = AutoTokenizer.from_pretrained('beomi/beep-KcELECTRA-base-hate')
    
    nh_dataloader = NhDataloader(
        nh_tokenizer,
        max_length=64
    )
    nh_inf_dataloader = nh_dataloader.get_dataloader(
        data,
        batch_size=128
    )
    return nh_inf_dataloader
    
def inference(nh_model, nh_classifier, nh_inf_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds = [] # 0: none, 1: hate ----df['label'] = preds

    with torch.no_grad():
        for _, data in enumerate(tqdm(nh_inf_dataloader, desc=f'Inference ')):
            encoded_anc_texts = data

            nh_encoded_texts = {
                "input_ids": encoded_anc_texts['input_ids'].to(device),
                "attention_mask": encoded_anc_texts['attention_mask'].to(device),
                "token_type_ids": encoded_anc_texts['token_type_ids'].to(device)
            }
            
            nh_outputs = nh_model(**nh_encoded_texts)

            nh_logit = nh_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            nh_logit = nh_logit.squeeze()
            
            pred = nh_classifier.predict(nh_logit)
            
            preds.extend(pred)
    
    return preds