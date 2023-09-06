import streamlit as st

text_input = st.text_input(
        "Enter some text 👇",
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

if text_input:
    st.write("You entered: ", text_input)