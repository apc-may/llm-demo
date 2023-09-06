import streamlit as st

text_input = st.text_input(
        "Enter some text 👇")

if text_input:
    st.write("You entered: ", text_input)