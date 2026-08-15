# Mental Health RAG Chatbot with Streamlit Interface

This project implements a conversational Retrieval Augmented Generation (RAG) Assistant focused on mental health topics. It utilizes a knowledge base stored in a Pinecone vector database, leverages Google's Gemini model for language understanding and generation, and provides a user-friendly chat interface built with Streamlit. The chatbot can understand conversational context, retrieve relevant information from its database, and stream responses back to the user.

## 🌟 Features

-   **Conversational AI**: Engages in multi-turn conversations, understanding context from chat history.
-   **Retrieval Augmented Generation (RAG)**:
    -   Utilizes a Pinecone vector database to store and retrieve relevant mental health Q&A data.
    -   Employs `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) for semantic understanding.
-   **Powered by Google Gemini**: Uses `ChatGoogleGenerativeAI` for:
    -   Rephrasing user queries based on chat history for better retrieval.
    -   Generating contextually relevant and informative answers based on retrieved data.
-   **Domain Specificity**: Designed to answer questions primarily within the mental health domain.
-   **Streaming Responses**: Answers are streamed token-by-token for an improved user experience.
-   **Interactive Web Interface**: Built with Streamlit, providing a clean and intuitive chat UI.
-   **Chat History Management**: Persists conversation history within the Streamlit session.
-   **Dynamic Index Creation**: Automatically creates the Pinecone index if it doesn't exist.

## ⚙️ Tech Stack

-   **Language**: Python 3.x
-   **Core Framework**: LangChain
-   **LLM**: Google Gemini (via `langchain-google-genai`)
-   **Vector Database**: Pinecone (via `pinecone-client`, `langchain-pinecone`)
-   **Embeddings**: Hugging Face Sentence Transformers (via `langchain-huggingface`)
-   **Web UI**: Streamlit
-   **Environment Management**: `python-dotenv`

## 🚀 Getting Started

### Prerequisites

-   Python 3.8 or higher
-   An active Google Cloud account with the Generative Language API enabled.
-   A Pinecone account.
-   A Hugging Face account (though `all-MiniLM-L6-v2` is a public model and typically doesn't require explicit token for download).

### Installation

1.  **Clone the repository:**
    ```
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` file should include:
    ```
    # Web Framework
    fastapi==0.115.0
    uvicorn[standard]==0.32.0
    
    # Core Dependencies
    pydantic>=2.11.1
    python-dotenv==1.0.0
    
    # LangChain
    langchain==0.3.7
    langchain-community==0.3.7
    langchain-pinecone==0.2.13
    langchain-huggingface==0.1.2
    langchain-google-genai==2.0.4
    
    # AI/ML
    google-generativeai==0.8.3
    pinecone-client==5.0.1
    transformers==4.46.3
    sentence-transformers==3.3.1

    ```
    Install them using:
    ```
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your API keys and Pinecone configuration:
    ```
    GOOGLE_API_KEY="your_google_generative_ai_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_CLOUD="your_pinecone_cloud_provider" # e.g., aws, gcp, azure
    PINECONE_REGION="your_pinecone_region"       # e.g., us-east-1
    PINECONE_INDEX_NAME="medical-chatbot"        
    HF_TOKEN="your_huggingface_token" 
    ```

### Data Preparation (If applicable)

-   If you have a dataset of mental health questions and answers, you'll need to process and upsert them into your Pinecone index (`PINECONE_INDEX_NAME`). The embeddings should be generated using `all-MiniLM-L6-v2`. This project currently assumes the pinecone index is ready to be populated/upserted with Data. jk, lost the script for upserting my dataset to pinecone. 

### Running the Application

1.  **Run the Streamlit web application (`streamlit_app.py`):**
    ```
    streamlit run streamlit_app.py
    ```
    This will start the Streamlit server, and you can interact with the Assistant in your web browser.

2.  **We have a deployed version. Do check it out.**
    ```
    https://mental-health-expert.streamlit.app/
    ```    

3. **Alternative: Run the command-line interface After navigating in the correct directory(`app.py`):**
    If you prefer a command-line version (as initially developed):
    ```
    python app.py
    ```
    This script will allow you to interact with the chatbot directly in your terminal.

## 🔧 How It Works

1.  **Initialization**:
    -   Loads API keys and environment variables.
    -   Initializes the embedding model (`all-MiniLM-L6-v2`).
    -   Connects to Pinecone, creating the specified index if it doesn't exist.
    -   Initializes the Google Gemini LLM (`ChatGoogleGenerativeAI`) with streaming enabled.

2.  **User Interaction (Streamlit App)**:
    -   The user types a question into the chat input.
    -   The current chat history and the new user question are sent to the RAG chain.

3.  **History-Aware Retriever**:
    -   The `create_history_aware_retriever` component takes the current question and chat history.
    -   It uses the LLM to rephrase the user's question into a standalone query, understandable without the full chat context. This is crucial for accurate retrieval in follow-up questions.

4.  **Document Retrieval**:
    -   The rephrased query is used by the `PineconeVectorStore` retriever to fetch the top `k` most semantically similar documents using cosine Similarity Metric (your Q&A data) from the Pinecone index.

5.  **Answer Generation**:
    -   The `create_stuff_documents_chain` takes the original user question, the chat history, and the retrieved documents (context).
    -   It uses another prompt and the LLM to synthesize an answer. The prompt guides the LLM to use the provided context, stay within the mental health domain, and be concise.

6.  **Streaming Response**:
    -   The generated answer is streamed back to the Streamlit interface, appearing token by token.

7.  **Chat History Update**:
    -   The user's question and the AI's full response are added to the session's chat history for future interactions.


