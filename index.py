import streamlit as st 
import numpy as np 
import json
import requests
from requests.exceptions import RequestException, ConnectionError, HTTPError, Timeout

#text_input = st.text_input(
#        "Enter some text 👇")

#if text_input:
#    st.write("You entered: ", text_input)
token = "dapi4462e69374f60947d518dfb63399c947"
#url = "https://adb-7621144643150231.11.azuredatabricks.net/driver-proxy-api/o/0/0908-093558-rn6h8ck8/7779"
url = "https://adb-7621144643150231.11.azuredatabricks.net/driver-proxy-api/o/0/0912-055511-23j1leuw/7779"
temperature=1.0 
max_new_tokens=1024

st.set_page_config(page_title='Demo', page_icon = "logo.jpg")
                        #, favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.title('Databricks Q&A bot')
#st.header('Databricks Q&A bot')

def generate_answer(question):
  # Driver Proxyと異なるクラスター、ローカルからDriver Proxyにアクセスする際にはパーソナルアクセストークンを設定してください
  headers = {
      "Content-Type": "application/json",
      "Authentication": f"Bearer {token}"
  }
  data = {
    "prompt": question,
    "temperature": temperature,
    "max_new_tokens": max_new_tokens,
  }

  response = requests.post(url, headers=headers, data=json.dumps(data))
  if response.status_code != 200:
    raise Exception(
       f"Request failed with status {response.status_code}, {response.text}"
    )
  
  response_json = response.json()
  return response_json

question = st.text_input("**質問**")

if question != "":
    response = requests.get(url)
    answer = generate_answer(question)

    st.write(f"**回答:** {answer}")

    #response.raise_for_status()