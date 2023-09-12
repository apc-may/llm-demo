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
import openai

# `Azure`固定
openai.api_type = "azure"
 
# Azure Open AI のエンドポイント
openai.api_base = "https://ka-abe-azureopen-api-japan-east.openai.azure.com/"
 
# Azure Docs 記載の項目
openai.api_version = "2023-05-15"
 
# Azure Open AI のキー
os.environ["OPENAI_API_KEY"] = answer
 
# デプロイ名
deployment_id = "ka-abe-gpt-turbo"
#deployment_id = "ka-abe-gpt-4"

# デプロイしたモデル名
model_name = "gpt-35-turbo"
#model_name = "gpt-4"

AgentAction = namedtuple('AgentAction', ['tool', 'tool_input', 'log'])

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

def run_agent(df):
    state = {"memory": ConversationBufferMemory(memory_key="chat_history")}
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, deployment_id=deployment_id), df, memory=state['memory'], verbose=True, return_intermediate_steps=True)
    prompt = """
    あなたはPythonでpandasのdataframeを操作しています。dataframeの名前は`df`です。
    あなたは以下のツールを使って、投げかけられた質問に日本語で答える必要があります：

    python_repl_ast： Pythonのシェルです。python_repl_ast：Pythonのシェルです。入力は有効なpythonコマンドである必要があります。このツールを使用すると、時々出力が省略されます - あなたの答えにそれを使用する前に、それが省略されたように見えないことを確認してください。

    与えられたデータから、以下の分析をステップバイステップで行ってください。

    Step.1 データの要約を説明する

    Step.2 基本統計量を確認する
    ここでは、平均値、中央値、標準偏差、最大値、最小値など、基本的な統計量を確認します。

    Step.3 データ分布の確認
    代表的なカラムを使って、散布図を描画してください。

    Step.4 分析のまとめと提案
    上記の結果を踏まえて、データの分析概要と、そこから得られる提案や仮説があれば提示してください。
    
    
    なお、質問と答えについては、次の形式を使用してください：

    質問：あなたが答えなければならない入力の質問
    思考：何をすべきか常に考えておくこと
    行動：取るべき行動。[python_repl_ast]のいずれかであるべきです。
    Action Input：アクションへの入力
    観察: アクションの結果
    ... （このThought/Action/Action Input/ObservationはN回繰り返すことができます。）
    思考： 最終的な答えがわかった
    最終的な答え：入力された元の質問に対する最終的な答え


    """
    result = agent({"input": prompt, "deployment_id":deployment_id})
    return result

st.title('Langchain Agent')

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください。", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    if st.button('実行'):
        with st.spinner('Agent is running...'):
            result = run_agent(df)
            answer = json.dumps(result['output'],ensure_ascii=False).replace('"', '')
            print(answer)
            st.write("分析結果:"+answer)
        
        actions = result['intermediate_steps']
        actions_list = []
        for action, result in actions:
            text = f"""Tool: {action.tool}\n
                Input: {action.tool_input}\n
                Log: {action.log}\nResult: {result}\n
            """
            if action.log is not None:
                st.write(action.log)
            if result is not None:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                if isinstance(result, matplotlib.collections.PathCollection):                    
                    st.pyplot()
                elif isinstance(result, matplotlib.axes.Axes):
                    st.pyplot()
                else:
                    st.write(result)

            
            text = re.sub(r'`[^`]+`', '', text)
            actions_list.append(text)
        
        with st.expander('ログ', expanded=False):
            st.write('\n'.join(actions_list))


        # st.write(answer)