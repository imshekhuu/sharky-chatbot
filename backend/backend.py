from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ══════════════════════════════════════════════════════════
# 1. LLM + EMBEDDINGS
# ══════════════════════════════════════════════════════════

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ══════════════════════════════════════════════════════════
# 2. PER-THREAD PDF STORE
# ══════════════════════════════════════════════════════════

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Chunk and embed a PDF into a per-thread FAISS store.
    Returns metadata: filename, pages, chunks.
    """
    if not file_bytes:
        raise ValueError("Empty file received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("PDF appears to be empty or unreadable.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        key = str(thread_id)
        _THREAD_RETRIEVERS[key] = retriever
        meta = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_METADATA[key] = meta
        return meta

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


# ══════════════════════════════════════════════════════════
# 3. TOOLS
# ══════════════════════════════════════════════════════════

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic arithmetic on two numbers.
    Supported operations: add, sub, mul, div.
    """
    ops = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else None,
    }
    if operation not in ops:
        return {"error": f"Unknown operation '{operation}'. Use: add, sub, mul, div."}

    if operation == "div" and second_num == 0:
        return {"error": "Division by zero is not allowed."}

    result = ops[operation](first_num, second_num)
    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a ticker symbol (e.g. 'AAPL', 'TSLA', 'INFY').
    Returns price, change, volume, and market cap when available.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol.upper().strip()}"
        "&apikey=C9PE94QUEW9VWGFM"
    )
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        quote = data.get("Global Quote", {})
        if not quote:
            return {"error": f"No data found for symbol '{symbol}'. Check the ticker."}

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "volume": quote.get("06. volume"),
            "latest_trading_day": quote.get("07. latest trading day"),
            "previous_close": quote.get("08. previous close"),
        }
    except requests.RequestException as e:
        return {"error": f"Network error: {str(e)}"}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Search the uploaded PDF document for information relevant to the query.
    Use this whenever the user asks about the document they uploaded.
    Always pass the thread_id so the correct document is searched.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this conversation. Ask the user to upload a PDF first.",
            "query": query,
        }

    results = retriever.invoke(query)
    if not results:
        return {
            "query": query,
            "context": [],
            "message": "No relevant sections found for this query.",
        }

    return {
        "query": query,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        "context": [doc.page_content for doc in results],
        "metadata": [doc.metadata for doc in results],
        "chunks_returned": len(results),
    }


tools: List = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# ══════════════════════════════════════════════════════════
# 4. SYSTEM PROMPT BUILDER
# ══════════════════════════════════════════════════════════

def _build_system(thread_id: Optional[str]) -> SystemMessage:
    has_doc = thread_has_document(str(thread_id)) if thread_id else False
    doc_name = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "the document") if thread_id else ""

    doc_instruction = (
        f"A PDF named '{doc_name}' is indexed for this conversation. "
        f"When the user asks about it, always call `rag_tool` with the thread_id `{thread_id}` "
        "and a clear, specific query string."
        if has_doc
        else "No document is currently uploaded. If the user asks about a PDF, tell them to upload one."
    )

    return SystemMessage(content=f"""You are Sharky, a sharp and capable AI assistant.

{doc_instruction}

Other tools available:
- `duckduckgo_results_json` — search the web for real-time information
- `get_stock_price` — look up live stock prices by ticker symbol
- `calculator` — perform arithmetic (add, sub, mul, div)

Guidelines:
- Be concise and precise. Avoid padding.
- Use tools proactively when they'd improve your answer.
- For RAG responses, cite the source document naturally.
- For stock prices, include the change % for context.
- If a tool returns an error, explain it clearly and suggest a fix.
""")


# ══════════════════════════════════════════════════════════
# 5. GRAPH STATE
# ══════════════════════════════════════════════════════════

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ══════════════════════════════════════════════════════════
# 6. NODES
# ══════════════════════════════════════════════════════════

def chat_node(state: ChatState, config=None) -> dict:
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system = _build_system(thread_id)
    messages = [system, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ══════════════════════════════════════════════════════════
# 7. CHECKPOINTER + GRAPH
# ══════════════════════════════════════════════════════════

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════
# 8. PUBLIC HELPERS
# ══════════════════════════════════════════════════════════

def generate_chat_title(first_message: str) -> str:
    """Generate a short 3–5 word title from the opening message."""
    try:
        response = llm.invoke(
            f"""Generate a short chat title for this message.

Message: {first_message}

Rules:
- 3 to 5 words
- No quotes or punctuation
- Sentence case
- Return the title only, nothing else
"""
        )
        title = response.content.strip()
        # safety: cap length in case the model returns more
        words = title.split()
        return " ".join(words[:6]) if words else first_message[:30]
    except Exception:
        return first_message[:30]


def retrieve_all_threads() -> list:
    """Return all thread IDs stored in the SQLite checkpointer."""
    all_threads: set = set()
    try:
        for ckpt in checkpointer.list(config={}):
            thread_id = ckpt.config.get("configurable", {}).get("thread_id")
            if thread_id:
                all_threads.add(thread_id)
    except Exception:
        pass
    return list(all_threads)