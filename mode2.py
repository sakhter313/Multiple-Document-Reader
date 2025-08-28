import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import tiktoken
from datetime import datetime
import pytz

# Inject custom CSS for chat-like styling
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(to bottom right, #e0f7fa, #ffffff);
        background-size: cover;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f4f8;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 5px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #DCF8C6;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #F0F0F0;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.title("📖 Document Reader Chatbot")
    st.markdown("""
    This chatbot uses RAG to provide exact answers from your uploaded documents.

    - **Upload**: PDF or TXT files.
    - **Index**: FAISS vector store.
    - **Chat**: Powered by Groq AI.
    - **Features**: Exact text matching, token usage, and comparison.
    """)
    models = {
        "llama-3.1-8b-instant": {"name": "LLaMA 3.1 8B", "max_tokens": 8192},
        "llama-3.3-70b-versatile": {"name": "LLaMA 3.3 70B", "max_tokens": 8192},
        "meta-llama/llama-guard-4-12b": {"name": "LLaMA Guard 4 12B", "max_tokens": 4096},
        "openai/gpt-oss-120b": {"name": "GPT-OSS 120B", "max_tokens": 16384},
        "openai/gpt-oss-20b": {"name": "GPT-OSS 20B", "max_tokens": 8192}
    }
    selected_model = st.selectbox("Select Groq Model", list(models.keys()), index=1)
    st.markdown(f"**Model Name**: {models[selected_model]['name']}")
    st.markdown(f"**Max Tokens**: {models[selected_model]['max_tokens']}")

# Function to estimate token count
def estimate_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return len(text.split()) * 2

# Function to process uploaded files and build vector store
@st.cache_resource
def build_vector_store(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                try:
                    loader = PyPDFLoader(tmp_path)
                except ImportError:
                    st.error("pypdf package not found. Please install it with `pip install pypdf`.")
                    continue
            elif file_extension == 'txt':
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                continue
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["file_name"] = uploaded_file.name
                doc.metadata["page"] = doc.metadata.get("page", len(docs) + 1)
            docs.extend(loaded_docs)
        finally:
            os.unlink(tmp_path)
    
    if not docs:
        raise ValueError("No valid documents uploaded.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Main Streamlit app
st.title("📚 Document Reader Chatbot")
st.markdown("Upload documents (PDF or TXT) and chat with the bot to get exact answers based on their content.")

# File uploader
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

# API Key from secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it in your app settings.")
else:
    groq_api_key = st.secrets["GROQ_API_KEY"]

    if uploaded_files:
        try:
            with st.spinner("Processing documents and building index..."):
                st.session_state.vectorstore = build_vector_store(tuple(uploaded_files))
            
            st.success("Documents processed! Ready to chat.")
            
            llm = ChatGroq(groq_api_key=groq_api_key, model=selected_model)
            system_prompt = (
                "You are a precise chatbot for answering questions based on uploaded documents. "
                "Return the exact text from the provided context when possible, limited to three sentences. "
                "If an exact match is not found, provide a concise summary based on the context, and state that it's a summary."
                "\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(
                st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
                question_answer_chain
            )
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
    
    # Chat interface
    st.markdown("### 💬 Chat with the Bot")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">👤 {message["content"]} <br><small>Tokens: {message["tokens"]} | {message["timestamp"]}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">🤖 {message["content"]} <br><small>Tokens: {message["tokens"]} | Model: {message["model"]} | {message["timestamp"]}</small></div>', unsafe_allow_html=True)
    
    if st.session_state.vectorstore and st.session_state.rag_chain:
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input("Your question:", placeholder="Ask about the documents...", key="chat_input")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and query:
                try:
                    with st.spinner("Generating answer..."):
                        response = st.session_state.rag_chain.invoke({"input": query})
                        answer = response["answer"]
                        context = response["context"]
                        
                        user_tokens = estimate_tokens(query)
                        bot_tokens = estimate_tokens(answer)
                        ist = pytz.timezone("Asia/Kolkata")
                        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": query,
                            "tokens": user_tokens,
                            "timestamp": timestamp
                        })
                        st.session_state.chat_history.append({
                            "role": "bot",
                            "content": answer,
                            "tokens": bot_tokens,
                            "model": models[selected_model]["name"],
                            "timestamp": timestamp
                        })
                        
                        # Display exact context for comparison
                        st.markdown("### 📑 Exact Text from Document")
                        for i, doc in enumerate(context, 1):
                            st.markdown(f"**Chunk {i} (Page {doc.metadata.get('page', 'N/A')}, File: {doc.metadata.get('file_name', 'unknown')})**:")
                            st.write(doc.page_content)
                            st.divider()
                        
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    else:
        st.info("Please upload documents to start chatting.")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
