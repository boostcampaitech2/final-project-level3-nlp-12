from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
# import 01-streamlit as st
from utils import read_json


# @st.cache(allow_output_mutation=True)
def get_pipeline():
    device: int = 0 if torch.cuda.is_available() else -1


    # model = torch.load(model_path, map_location=device)
    model = AutoModelForSequenceClassification.from_pretrained("beomi/beep-KcELECTRA-base-hate")

    model_config = AutoConfig.from_pretrained('beomi/beep-KcELECTRA-base-hate')

    tokenizer = AutoTokenizer.from_pretrained('beomi/beep-KcELECTRA-base-hate')

    pipe = pipeline(
        task="text-classification",
        config=model_config,
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return pipe