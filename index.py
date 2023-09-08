import streamlit as st 
import numpy as np 
import json
import requests
from requests.exceptions import RequestException, ConnectionError, HTTPError, Timeout

#text_input = st.text_input(
#        "Enter some text 👇")

#if text_input:
#    st.write("You entered: ", text_input)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
url = "http://127.0.0.1:7777/"


st.title('Databricks Q&A bot')
#st.header('Databricks Q&A bot')

def generate_answer(question):
  # Driver Proxyと異なるクラスター、ローカルからDriver Proxyにアクセスする際にはパーソナルアクセストークンを設定してください
  headers = {
      "Content-Type": "application/json",
      "Authentication": f"Bearer {token}"
  }
  data = {
    "prompt": question
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
    answer = generate_answer(question)

    answer = answer["answer"]
    source = answer["source"]

    st.write(f"**回答:** {answer}")
    st.write(f"**ソース:** [{source}]({source})")

    #response.raise_for_status()
  except ConnectionError as ce:
    print("Connection Error:", ce)
  except HTTPError as he:
    print("HTTP Error:", he)
  except Timeout as te:
    print("Timeout Error:", te)
  except RequestException as re:
    print("Error:", re)