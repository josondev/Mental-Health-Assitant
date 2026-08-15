import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_pinecone import PineconeVectorStore

# Page config
st.set_page_config(
    page_title="Mental Health Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 10% 0%, #1d4ed8 0, transparent 35%), radial-gradient(circle at 90% 10%, #7c3aed 0, transparent 32%), #080b14; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 7rem; }
.hero { padding: 28px 30px; border-radius: 24px; margin-bottom: 22px; background: linear-gradient(135deg, rgba(37,99,235,.30), rgba(124,58,237,.25)); border: 1px solid rgba(255,255,255,.12); box-shadow: 0 18px 60px rgba(0,0,0,.25); }
.hero h1 { margin: 0; color: #fff; font-size: 2.25rem; letter-spacing: -.04em; }
.hero p { margin: 8px 0 0; color: #cbd5e1; font-size: 1rem; }
.status-pill { display:inline-block; padding:5px 11px; border-radius:999px; background:rgba(255,255,255,.10); color:#e2e8f0; font-size:.78rem; border:1px solid rgba(255,255,255,.12); }
section[data-testid="stSidebar"] { background: rgba(8,11,20,.96); border-right: 1px solid rgba(255,255,255,.08); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#f8fafc; }
div[data-testid="stChatMessage"] { border: 1px solid rgba(255,255,255,.09); border-radius: 20px; padding: 1.15rem 1.25rem; margin: .8rem 0; background: rgba(15,23,42,.76); box-shadow: 0 10px 30px rgba(0,0,0,.18); }
div[data-testid="stChatMessageContent"] { color:#e5e7eb; }
div[data-testid="stChatMessageContent"] h1, div[data-testid="stChatMessageContent"] h2, div[data-testid="stChatMessageContent"] h3 { color:#fff; margin-top:.5rem; }
div[data-testid="stChatMessageContent"] table { border-radius:12px; overflow:hidden; }
div[data-testid="stChatMessageContent"] code { color:#c4b5fd; }
div[data-testid="stExpander"] { border:1px solid rgba(255,255,255,.09); border-radius:14px; }
.small-muted { color:#94a3b8; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def initialize_chatbot():
    """Initialize the RAG chatbot (cached to avoid reloading)"""
    try:
        # Get API keys from Streamlit secrets
        pinecone_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
        google_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        pinecone_cloud = st.secrets.get("PINECONE_CLOUD", os.getenv("PINECONE_CLOUD", "aws"))
        pinecone_region = st.secrets.get("PINECONE_REGION", os.getenv("PINECONE_REGION", "us-east-1"))

        if not pinecone_key or not google_key:
            st.error("⚠️ API keys not found. Please add them to Streamlit secrets.")
            st.stop()

        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_key)
        spec = ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Pinecone index
        index_name = "medical-chatbot"
        try:
            pc.Index(index_name).describe_index_stats()
        except PineconeApiException:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=spec
            )

        # Vector store
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.1,
            max_tokens=1024,
            api_key=google_key,
        )

        # Contextualize question prompt
        contextualize_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualized_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=base_retriever,
            prompt=contextualized_q_prompt,
        )

        # QA prompt
        system_prompt = (
            "You are a compassionate mental health assistant. "
            "Use the following context to answer the question. "
            "Focus ONLY on mental health topics. "
            "If you don't know, say so. Be empathetic and supportive."
            "\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Final RAG chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain,
        )

        return rag_chain

    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None

# Initialize chatbot
if st.session_state.rag_chain is None:
    with st.spinner("🔄 Loading AI assistant..."):
        st.session_state.rag_chain = initialize_chatbot()

# Header
st.title("🧠 Mental Health Assistant")
st.caption("Your compassionate AI companion for mental health support")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI assistant provides mental health support using:
    - **RAG** (Retrieval Augmented Generation)
    - **Google Gemini** for natural language understanding
    - **Pinecone** for knowledge retrieval

    💡 **Tips:**
    - Ask questions about stress, anxiety, depression
    - The AI remembers your conversation context
    - This is not a replacement for professional help
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption("⚠️ **Disclaimer:** This is an AI assistant. For emergencies, contact professional help immediately.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Check if chatbot is initialized
    if st.session_state.rag_chain is None:
        st.error("⚠️ Chatbot not initialized. Please check your API keys.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke RAG chain
                result = st.session_state.rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })

                response = result.get("answer", "I'm sorry, I couldn't generate a response.")

                # Display response
                st.markdown(response)

                # Update chat history
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=response))

                # Add to messages
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
