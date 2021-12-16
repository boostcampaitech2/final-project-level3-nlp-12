from datetime import date
import streamlit as st
import yaml
from predict import get_pipeline

from confirm_button_hack import cache_on_button_press
from load_data import *
from service.api_response import *
from service.error_handler import *

import logging

st.set_page_config(layout='wide')
st.header('Hello AI-it!!')

st.title('Malicious Comments Collecting Service')

def main():    
    st.write('Model Loading...')
    pipe = get_pipeline()
    
    keyword = st.text_input('Keyword you want to collect!!')    
    comments = retrieve_comments(keyword)
    
    results = []
    try:
        for comment in comments[:10]:
            output = pipe(comment)
            results.append(
                {
                    'comment':comment, 
                    'label':output[0]['label'], 
                    'score':output[0]['score']
                }
            )
        st.write(ApiResponse.success(Status.SUCCESS, Msg.SUCCESS, results, datetime.now().strftime('%Y/%m/%d, %H:%M:%S')))
    except:
        st.write(ApiResponse.fail(Status.BAD_REQUEST, Msg.WRONG_FORMAT, results, datetime.now().strftime('%Y/%m/%d, %H:%M:%S')))


root_password = '123'
password = st.text_input('password', type='password')

@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    st.write(type(password))
    return password == root_password

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')