# alfred_streamlit.py
import streamlit as st
from langchain_core.messages import HumanMessage
from tools import alfred  # Import your LangGraph agent

st.set_page_config(page_title="🕵️ Alfred - AI Assistant", page_icon="🎩")

st.title("🎩 Alfred - Your AI Assistant")
st.markdown("Ask Alfred anything. He’s connected to weather, search, model stats, and even your guest list!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask Alfred...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.spinner("Alfred is thinking..."):
        response = alfred.invoke({"messages": st.session_state.chat_history})
        ai_response = response['messages'][-1].content
        st.session_state.chat_history.append(response['messages'][-1])

# Display chat history
for msg in st.session_state.chat_history:
    role = "🤵 Alfred" if msg.type == "ai" else "🧑 You"
    st.chat_message(role).markdown(msg.content)
