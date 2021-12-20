import streamlit as st
from confirm_button_hack import cache_on_button_press
import requests
import time
import pandas as pd
import json
from collections import defaultdict
from PIL import Image
from io import BytesIO


st.set_page_config(layout='wide')
st.header('Hello AI-it!!')

st.title('Malicious Comments Collecting Service')


def main():
    keyword = st.text_input('Keyword you want to collect!!')
    if keyword:

        # st.write('Classifying...')
        with st.spinner('Collecting Evidence...'):
            # local
            response = requests.get('http://localhost:8000/inference/' + keyword)
            # ip
            # response = requests.get('http://35.185.204.191/inference/' + keyword)

        st.success('Done!')

        st.markdown("<h2 style='text-align: center'>Report</h1>", unsafe_allow_html=True)
        st.markdown('-----------------------')
        st.markdown('#### Label Description')
        st.markdown('- **Hate**     : 혐오적인 표현')
        st.markdown('- **Offensive**: 공격적인 표현')
        st.markdown('-----------------------')

        col1, col2 = st.columns(2)

        jsonfile = response.json()
        with col1:
            for i, res in enumerate(response.json()[:5]):
                st.subheader(f'Evidence:{i+1}')
                st.markdown("comment: "+res["comment"])
                #st.write(res)

            with st.expander("Detail Information"):
                df = pd.DataFrame(jsonfile)
                st.write(df)

        with col2:
            # wordcloud
            st.subheader('Word Cloud')
            newtext = json_to_text(jsonfile)
            response = requests.post('http://localhost:8000/wordcloud/', json = {'comment': newtext, 'keyword': keyword})
            #response = requests.post('http://localhost:8000/wordcloud/', data = newtext)

            image = Image.open(BytesIO(response.content))

            st.image(image, caption='wordcloud')

            st.markdown('-----------------------')
            st.subheader('통계')
            df = pd.DataFrame(jsonfile)
            st.bar_chart(df.groupby('label').count()['score'])
            st.markdown('-----------------------')
            st.subheader('파일 다운로드')


            csv = json_to_csv(jsonfile)
            st.download_button(
                label="Download data as CSV",
                    data = csv,
                    file_name='final data.csv',
                    mime='text/csv'
                )


root_password = '123'
password = st.text_input('password', type='password')

@st.cache
def json_to_text(json):
    newtext = ""

    for info in json:
        newtext += info['comment']
        newtext += " "
    return newtext

@st.cache
def json_to_csv(json):
    df = pd.json_normalize(json)
    return df.to_csv().encode('utf-8')

@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password


if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')