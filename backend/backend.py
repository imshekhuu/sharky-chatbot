from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# --------------------------------------------------
# Chat Title Generator
# --------------------------------------------------

def generate_chat_title(first_message: str) -> str:
    try:
        response = llm.invoke(
            f"""
            Generate a short chat title.

            User Message:
            {first_message}

            Rules:
            - 3 to 5 words
            - No quotes
            - No punctuation
            - Return title only
            """
        )

        return response.content.strip()

    except Exception:
        return first_message[:30]


# --------------------------------------------------
# LangGraph State
# --------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


checkpoint = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)