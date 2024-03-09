import streamlit as st
import pandas as pd
import asyncio
from streamlit_chat import message
from main import conversational_chat, setup_chain

st.set_page_config(layout="wide", page_icon="💬", page_title="Essential-ChatBot")

st.markdown(
    "<h1 style='text-align: center;'>Essential-ChatBot, Talk with your CSV data!</h1>",
    unsafe_allow_html=True)

user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")

def show_user_file(uploaded_file):
    file_container = st.expander("Your CSV file :")
    shows = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
    file_container.write(shows)

def main():
    if user_api_key == "":
        st.markdown(
            "<div style='text-align: center;'><h4>Enter your OpenAI API key to start chatting 😉</h4></div>",
            unsafe_allow_html=True)
        return
    
    uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="hidden")
    if uploaded_file is not None:
        show_user_file(uploaded_file)
    else:
        st.sidebar.info("👆 Upload your CSV file to get started.")
        return
    
    with st.sidebar.expander("🛠️ Settings", expanded=False):
        if st.button("Reset Chat"):
            st.session_state['reset_chat'] = True

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'ready' not in st.session_state:
        st.session_state['ready'] = False
    if 'reset_chat' not in st.session_state:
        st.session_state['reset_chat'] = False

    if uploaded_file is not None:
        chain = setup_chain(uploaded_file, user_api_key)
        st.session_state['ready'] = True

    if st.session_state['ready']:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]
        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

        response_container = st.container()
        with st.container():
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

                if st.session_state['reset_chat']:
                    st.session_state['history'] = []
                    st.session_state['past'] = ["Hey! 👋"]
                    st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]
                    response_container.empty()
                    st.session_state['reset_chat'] = False

            if submit_button and user_input:
                SYSTEM_PROMPT="""
                                """
                output = conversational_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

about = st.sidebar.expander("About 🤖")

if __name__ == "__main__":
    main()
