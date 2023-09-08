import streamlit as st 
import numpy as np 
import json
import requests
from requests.exceptions import RequestException, ConnectionError, HTTPError, Timeout

#text_input = st.text_input(
#        "Enter some text 👇")

#if text_input:
#    st.write("You entered: ", text_input)
token = "dapib7d7c12712fb4730923210d5425d054f-3"
url = "http://10.139.64.4:7779/"
temperature=1.0 
max_new_tokens=1024

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
  try:
    response = requests.get(url)
    st.write(response)
    answer = generate_answer(question)

    #answer = answer["answer"]
    #source = answer["source"]

    st.write(f"**回答:** {answer}")
    #st.write(f"**ソース:** [{source}]({source})")

    #response.raise_for_status()
  except ConnectionError as ce:
    st.write("Connection Error:",ce)
  except HTTPError as he:
    st.write("HTTP Error:",he)
  except Timeout as te:
    st.write("Timeout Error:", te)
  except RequestException as re:
    st.write("Error:", re)