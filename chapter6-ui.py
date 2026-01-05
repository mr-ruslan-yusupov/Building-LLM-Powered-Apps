import streamlit as st
import uuid
from dotenv import load_dotenv
import os

# 1. Modern LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool

# 2. Modern LangGraph Imports
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

# --- Page & App Setup ---
load_dotenv()
st.set_page_config(page_title="GlobeBotter", page_icon="🌐")
st.header('Welcome to GlobeBotter, your travel assistant with Internet access.')
st.subheader('What are you planning for your next trip?')

# --- Caching Setup Functions ---
# Use Streamlit's caching to avoid re-initializing on every interaction.

@st.cache_resource
def setup_retriever():
    """Create and cache the document retriever."""
    try:
        if not os.path.exists('Travel_Germany.pdf'):
            st.error("Travel_Germany.pdf not found. Please add the file and restart.")
            return None
        loader = PyPDFLoader('Travel_Germany.pdf')
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)
        db = FAISS.from_documents(documents, OpenAIEmbeddings())
        return db.as_retriever()
    except Exception as e:
        st.error(f"Error setting up the retriever: {e}")
        return None

@st.cache_resource
def setup_tools(_retriever):
    """Create and cache the tools for the agent."""
    if _retriever is None:
        return []
    search = SerpAPIWrapper()
    retriever_tool = create_retriever_tool(
        _retriever,
        "Travel_Germany_Information",
        "Searches and returns documents regarding travel in Germany."
    )
    search_tool = Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events, weather, or general knowledge not related to Germany travel."
    )
    return [retriever_tool, search_tool]

@st.cache_resource
def setup_agent(_tools):
    """Create and cache the LangGraph agent."""
    if not _tools:
        return None
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    memory = MemorySaver()
    return create_agent(llm, _tools, checkpointer=memory)


# --- Initialize Agent and Tools ---
retriever = setup_retriever()
tools = setup_tools(retriever)
agent_executor = setup_agent(tools)

# --- Session State Management ---
# Manage chat history and conversation ID (thread_id)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- UI and Chat Logic ---

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if user_query := st.chat_input(placeholder="Ask me anything about travel!"):
    if not agent_executor:
        st.error("The agent is not initialized. Please check your setup and API keys.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # This is the modern replacement for StreamlitCallbackHandler
        with st.chat_message("assistant"):
            # The placeholder for streaming output
            message_placeholder = st.empty()
            full_response = ""

            # Define the config for this specific conversation thread
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Stream the agent's response
            stream = agent_executor.stream({"messages": [("user", user_query)]}, config=config)

            for chunk in stream:
                # The final answer is in the 'messages' key of the last event
                # Chunk structure is like: {'agent': {'messages': [...]}} or {'tools': {'messages': [...]}}
                for node, values in chunk.items():
                    if "messages" in values:
                        last_msg = values["messages"][-1]

                        # We only want to display the message if:
                        # 1. It is an AIMessage (not a ToolMessage/Search Result)
                        # 2. It actually has text content (not just a Tool Call)
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            full_response = last_msg.content
                            # Add a blinking cursor to simulate typing
                            message_placeholder.markdown(full_response + "▌")

            # Display the final message without the cursor
            message_placeholder.markdown(full_response)

        # Add the final response to the message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar for Resetting Chat ---
if st.sidebar.button("Reset chat history"):
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    # Start a new conversation by generating a new thread_id
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()