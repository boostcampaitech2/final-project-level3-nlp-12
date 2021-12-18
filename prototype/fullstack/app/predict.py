from transformers import pipeline, AutoTokenizer
import torch
# import streamlit as st
from utils import read_json

# @st.cache(allow_output_mutation=True)
def get_pipeline():
    print('here@@@@@@@@@@@@@@@@@')
    device = 0 if torch.cuda.is_available() else -1

    model_path = '/Users/yangjaeug/Desktop/GitHub/Product-Serving/assets/comments_task/model.pt'
    config_path = '/Users/yangjaeug/Desktop/GitHub/Product-Serving/practice/st_practice/config.json'

    print('!!!')
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print('???')
    model_config = read_json(config_path)

    tokenizer = AutoTokenizer.from_pretrained('beomi/KcELECTRA-base')

    pipe = pipeline(
        task="text-classification",
        config=model_config,
        model=model.model,
        tokenizer=tokenizer,
        device=device
    )
    
    return pipe