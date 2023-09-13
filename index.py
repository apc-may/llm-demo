import streamlit as st 
import numpy as np 
import json
import requests
from requests.exceptions import RequestException, ConnectionError, HTTPError, Timeout
import pandas as pd
import plotly.figure_factory as ff

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
#st.title('Databricks Q&A bot')
#st.header('Databricks Q&A bot')

#def generate_answer(question):
  # Driver Proxyと異なるクラスター、ローカルからDriver Proxyにアクセスする際にはパーソナルアクセストークンを設定してください
#  headers = {
#      "Content-Type": "application/json",
#      "Authentication": f"Bearer {token}"
#  }
#  data = {
#    "prompt": question,
#    "temperature": temperature,
#    "max_new_tokens": max_new_tokens,
#  }

#  response = requests.post(url, headers=headers, data=json.dumps(data))
#  if response.status_code != 200:
#    raise Exception(
#       f"Request failed with status {response.status_code}, {response.text}"
#    )
  
#  response_json = response.json()
#  return response_json

#question = st.text_input("**質問**")

#if question != "":
#    response = requests.get(url)
#    answer = generate_answer(question)

#    st.write(f"**回答:** {answer}")


#Langchain test#
import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage
from typing import Any, Dict, List
import pandas as pd
import os
import json
import re
from collections import namedtuple
import matplotlib.collections
import matplotlib.pyplot as plt
#import openai

# `Azure`固定
#openai.api_type = "azure"
 
# Azure Open AI のエンドポイント
#openai.api_base = "https://ka-abe-azureopen-api-japan-east.openai.azure.com/"
 
# Azure Docs 記載の項目
#openai.api_version = "2023-05-15"
 
# Azure Open AI のキー
os.environ["OPENAI_API_KEY"] =  st.secrets['path']
os.environ["OPENAI_ORGANIZATION"] = "org-tD1A9K2bGhfzsjXS9RGHyfdd"
#openai.api_key = os.getenv("OPENAI_API_KEY")
#openai.organization = "org-tD1A9K2bGhfzsjXS9RGHyfdd"
# デプロイ名
#deployment_id = "ka-abe-gpt-turbo"
#deployment_id = "ka-abe-gpt-4"

# デプロイしたモデル名
model_name = "gpt-35-turbo"
#model_name = "gpt-4"

from streamlit_chat import message
import pexpect
import json
import re
from collections import namedtuple

# From here down is all the StreamLit UI.
#st.set_page_config(page_title="📊 ChatCSV", page_icon="📊")
#st.header("📊 ChatCSV")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []
    
    
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from typing import Any, Dict, List

df = pd.DataFrame([])
data = st.file_uploader(label='Upload CSV file', type='csv')

# st.download_button(label='サンプルデータをダウンロードする',data='https://drive.google.com/file/d/1wuSx35y3-hjZew1XhrM78xlAGIDTd4fp/view?usp=drive_open',mime='text/csv')

#header_num = st.number_input(label='Header position',value=0)
#index_num = st.number_input(label='Index position',value=2)
header_num = 0
index_num = 1
index_list = [i for i in range(index_num)]

if data:
    df = pd.read_csv(data,header=header_num,index_col=index_list)
    st.dataframe(df)

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

ask_button = ""
language = st.selectbox('language',['English','日本語'])

if df.shape[0] > 0:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, max_tokens=1000), df, memory=state['memory'], verbose=True, return_intermediate_steps=True, max_iterations=10, max_execution_time=40)
    user_input = get_text()
    ask_button = st.button('ask')
else:
    pass

AgentAction = namedtuple('AgentAction', ['tool', 'tool_input', 'log'])

def format_action(action, result):
    action_fields = '\n'.join([f"{field}: {getattr(action, field)}"+'\n' for field in action._fields])
    return f"{action_fields}\nResult: {result}\n"

if ask_button:
    st.write("Input:", user_input)
    with st.spinner('typing...'):
        prefix = f'You are the best explainer. please answer in {language}. User: '
        handler = SimpleStreamlitCallbackHandler()
        response = agent({"input":user_input})
        
        
        actions = response['intermediate_steps']
        actions_list = []
        for action, result in actions:
            text = f"""Tool: {action.tool}\n
               Input: {action.tool_input}\n
               Log: {action.log}\nResult: {result}\n
            """
            text = re.sub(r'`[^`]+`', '', text)
            actions_list.append(text)
            
        answer = json.dumps(response['output'],ensure_ascii=False).replace('"', '')
        if language == 'English':
            with st.expander('ℹ️ Show details', expanded=False):
                st.write('\n'.join(actions_list))
        else:
            with st.expander('ℹ️ 詳細を見る', expanded=False):
                st.write('\n'.join(actions_list))
            
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
        
if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")