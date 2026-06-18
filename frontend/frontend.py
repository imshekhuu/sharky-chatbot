import os
import sys
import uuid

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from backend.backend import chatbot, generate_chat_title

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def generate_id():
    return str(uuid.uuid4())


def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():

    new_thread_id = generate_id()

    st.session_state["thread_id"] = new_thread_id

    add_thread(new_thread_id)

    st.session_state["message_history"] = []


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

add_thread(st.session_state["thread_id"])

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🦈 Sharky Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.markdown("---")
st.sidebar.subheader("Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:

    title = st.session_state["chat_titles"].get(
        thread_id,
        "New Chat"
    )

    if st.sidebar.button(
        f"💬 {title}",
        key=f"thread_{thread_id}"
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            role = (
                "user"
                if isinstance(msg, HumanMessage)
                else "assistant"
            )

            temp_messages.append(
                {
                    "role": role,
                    "content": msg.content
                }
            )

        st.session_state["message_history"] = temp_messages


# --------------------------------------------------
# Main Chat Window
# --------------------------------------------------

st.title("🦈 Sharky Chatbot")

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

user_input = st.chat_input("Ask anything...")

if user_input:

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Create title on first message
    if (
        st.session_state["thread_id"]
        not in st.session_state["chat_titles"]
    ):

        title = generate_chat_title(user_input)

        st.session_state["chat_titles"][
            st.session_state["thread_id"]
        ] = title

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):

        def ai_stream():

            for chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):

                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_response = st.write_stream(ai_stream())

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    st.rerun()