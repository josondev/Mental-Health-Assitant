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

# 1. Page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Mental Health Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS Block
st.markdown("""
<style>
/* 1. Global Background */
.stApp { 
    background: radial-gradient(circle at 10% 0%, #1d4ed8 0, transparent 40%), 
                radial-gradient(circle at 90% 10%, #7c3aed 0, transparent 35%), 
                #080b14 !important; 
}

/* 2. Target ALL Streamlit container variations to force center alignment & top spacing */
.main .block-container,
[data-testid="stMainBlockContainer"],
.stMainBlockContainer { 
    max-width: 850px !important; 
    padding-top: 5rem !important;
    padding-bottom: 6rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* 3. Ensure the app layout centers content cleanly */
[data-testid="stAppViewContainer"] > .main {
    display: flex !important;
    justify-content: center !important;
}

/* 4. Custom Header Card Styling */
.custom-header {
    text-align: center !important;
    padding: 2.5rem 2rem !important;
    margin: 0 auto 2.5rem auto !important;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(124, 58, 237, 0.2)) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.35) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.custom-header h1 {
    color: #ffffff !important;
    font-size: 2.3rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.6rem 0 !important;
    padding: 0 !important;
    letter-spacing: -0.02em !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
}

.custom-header p {
    color: #cbd5e1 !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 5. Chat Messages Styling */
div[data-testid="stChatMessage"] { 
    border: 1px solid rgba(255, 255, 255, 0.12) !important; 
    border-radius: 16px !important; 
    padding: 1.2rem 1.4rem !important; 
    margin: 1rem 0 !important; 
    background: rgba(15, 23, 42, 0.85) !important; 
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important; 
}

div[data-testid="stChatMessageContent"] { 
    color: #f8fafc !important; 
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

/* 6. Chat Input Styling */
div[data-testid="stChatInput"] textarea {
    color: #ffffff !important;
    background-color: rgba(15, 23, 42, 0.9) !important;
}

/* 7. Sidebar text color fix */
section[data-testid="stSidebar"] { 
    background: rgba(8, 11, 20, 0.96) !important; 
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important; 
}

section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] li {
    color: #cbd5e1 !important;
}
</style>
""", unsafe_allow_html=True)

# 3. Render Custom Centered Hero Header
st.markdown("""
<div class="custom-header">
    <h1>🧠 Mental Health Assistant</h1>
    <p>Your compassionate AI companion for mental health support</p>
</div>
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
        # Get API keys from Streamlit secrets or env vars
        pinecone_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
        google_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        pinecone_cloud = st.secrets.get("PINECONE_CLOUD", os.getenv("PINECONE_CLOUD", "aws"))
        pinecone_region = st.secrets.get("PINECONE_REGION", os.getenv("PINECONE_REGION", "us-east-1"))

        if not pinecone_key or not google_key:
            st.error("⚠️ API keys not found. Please add them to Streamlit secrets or environment variables.")
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
            model="gemini-1.5-flash",
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

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI assistant provides mental health support using:
    - **RAG** (Retrieval Augmented Generation)
    - **Google Gemini** for natural language understanding
    - **Pinecone** for knowledge retrieval

    💡 **Tips:**
    - Ask questions about stress, anxiety, or coping strategies.
    - The AI remembers your conversation context.
    - This is not a replacement for professional medical advice.
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption("⚠️ **Disclaimer:** This is an AI assistant. For emergencies, please contact a local crisis hotline or mental health professional immediately.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
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
