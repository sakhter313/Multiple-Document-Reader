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

# Enhanced custom CSS for a more modern and attractive look
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #f8f9ff, #e8eaf6);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    .chat-message {
        padding: 15px;
        border-radius: 20px;
        margin: 10px 0;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(45deg, #DCF8C6, #AED581);
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background: linear-gradient(45deg, #E3F2FD, #BBDEFB);
        margin-right: auto;
    }
    .pdf-data {
        background: linear-gradient(to bottom, #f9f9f9, #e0e0e0);
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 10px;
        max-height: 400px;
        overflow-y: auto;
        margin-top: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(45deg, #ffebee, #ffcdd2);
        padding: 15px;
        border: 1px solid #ef9a9a;
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(45deg, #e8f5e8, #c8e6c9);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .doc-list {
        background: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content with enhanced layout
with st.sidebar:
    st.title("📖 Document Reader Chatbot")
    st.markdown("""
    <div class="info-box">
    This chatbot uses RAG to provide exact answers from your uploaded documents. Enhanced with modern UI for better experience!
    <ul>
        <li><strong>Upload</strong>: PDF or TXT files.</li>
        <li><strong>Index</strong>: FAISS vector store.</li>
        <li><strong>Chat</strong>: Powered by Groq AI.</li>
        <li><strong>New Features</strong>: Document stats, full text view, improved chat.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    models = {
        "llama-3.1-8b-instant": {"name": "LLaMA 3.1 8B", "max_tokens": 8192},
        "llama-3.3-70b-versatile": {"name": "LLaMA 3.3 70B", "max_tokens": 8192},
        "meta-llama/llama-guard-4-12b": {"name": "LLaMA Guard 4 12B", "max_tokens": 4096},
        "openai/gpt-oss-120b": {"name": "GPT-OSS 120B", "max_tokens": 16384},
        "openai/gpt-oss-20b": {"name": "GPT-OSS 20B", "max_tokens": 8192}
    }
    selected_model = st.selectbox("Select Groq Model", [k for k in models.keys()], index=1, help="Choose the AI model for responses.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model**: {models[selected_model]['name']}")
    with col2:
        st.markdown(f"**Max Tokens**: {models[selected_model]['max_tokens']}")

# Function to estimate token count (unchanged)
def estimate_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return len(text.split()) * 2

# Enhanced function to process uploaded files and build vector store with stats
@st.cache_resource
def build_vector_store(uploaded_files):
    docs = []
    full_text = ""
    total_pages = 0
    total_chars = 0
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["file_name"] = uploaded_file.name
                    doc.metadata["page"] = doc.metadata.get("page", total_pages + 1)
                    docs.append(doc)
                    full_text += doc.page_content + "\n"
                    total_pages += 1
                    total_chars += len(doc.page_content)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["file_name"] = uploaded_file.name
                    docs.append(doc)
                    full_text += doc.page_content + "\n"
                    total_chars += len(doc.page_content)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                continue
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
        finally:
            os.unlink(tmp_path)
    
    if not docs:
        raise ValueError("No valid documents uploaded.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Calculate total tokens approx
    total_tokens = estimate_tokens(full_text)
    
    return vectorstore, full_text, {"pages": total_pages, "chars": total_chars, "tokens": total_tokens, "files": len(uploaded_files)}

# Initialize session state (unchanged)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_full_text" not in st.session_state:
    st.session_state.pdf_full_text = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "processing_errors" not in st.session_state:
    st.session_state.processing_errors = []

# Main Streamlit app with enhanced layout
st.title("📚 Enhanced Document Reader Chatbot")
st.markdown("**Upload your PDF or TXT documents below and start chatting to extract precise insights!** 🚀")

# Use columns for better layout
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True, help="Select multiple PDF or TXT files to analyze.")
with col2:
    if st.button("🗑️ Clear All", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.pdf_full_text = ""
        st.session_state.doc_stats = {}
        st.session_state.processing_errors = []
        st.rerun()

# API Key check
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ **GROQ_API_KEY** not found in Streamlit secrets. Please add it in your app settings.")
    st.stop()

groq_api_key = st.secrets["GROQ_API_KEY"]

if uploaded_files:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Processing documents...")
        progress_bar.progress(50)
        
        st.session_state.vectorstore, st.session_state.pdf_full_text, st.session_state.doc_stats = build_vector_store(tuple(uploaded_files))
        progress_bar.progress(100)
        status_text.text("Done! Building AI chain...")
        
        st.session_state.processing_errors = []  # Clear errors
        
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
        
        st.success("✅ **Documents processed successfully!** Ready to chat.")
        
        # Display document stats and list
        st.markdown("### 📊 Document Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files", st.session_state.doc_stats["files"])
        with col2:
            st.metric("Pages", st.session_state.doc_stats["pages"])
        with col3:
            st.metric("Tokens", f"{st.session_state.doc_stats['tokens']:,}")
        
        # List of uploaded files (simplified)
        st.markdown("### 📋 Uploaded Files")
        for f in uploaded_files:
            st.markdown(f'<div class="doc-list">📄 {f.name}</div>', unsafe_allow_html=True)
        
        # Expander for full text view
        with st.expander("👁️ View Full Document Text", expanded=False):
            st.markdown('<div class="pdf-data">', unsafe_allow_html=True)
            st.text_area("", st.session_state.pdf_full_text, height=300, disabled=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ **Error building index**: {str(e)}")
        st.session_state.processing_errors.append(str(e))

# Display processing errors
if st.session_state.processing_errors:
    st.markdown("### ⚠️ Processing Errors")
    for error in st.session_state.processing_errors:
        st.markdown(f'<div class="error-box">{error}</div>', unsafe_allow_html=True)

# Enhanced Chat interface using native Streamlit chat elements
if st.session_state.vectorstore and st.session_state.rag_chain:
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Documents")
    
    # Display chat history using st.chat_message
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(f"Tokens: {message['tokens']} | {message['timestamp']}" + (f" | Model: {message.get('model', '')}" if message["role"] == "bot" else ""))
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        with st.chat_message("user"):
            st.markdown(prompt)
            user_tokens = estimate_tokens(prompt)
            ist = pytz.timezone("Asia/Kolkata")
            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt,
                "tokens": user_tokens,
                "timestamp": timestamp
            })
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    context = response["context"]
                    
                    bot_tokens = estimate_tokens(answer)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "tokens": bot_tokens,
                        "model": models[selected_model]["name"],
                        "timestamp": timestamp
                    })
                    
                    st.markdown(answer)
                    st.caption(f"Tokens: {bot_tokens} | Model: {models[selected_model]['name']} | {timestamp}")
                    
                    # Display exact context
                    with st.expander("📑 View Source Context", expanded=True):
                        for i, doc in enumerate(context, 1):
                            st.markdown(f"**Chunk {i}** - Page {doc.metadata.get('page', 'N/A')} | File: {doc.metadata.get('file_name', 'unknown')}")
                            st.write(doc.page_content)
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    st.session_state.processing_errors.append(str(e))

else:
    # Welcome message if no documents
    st.info("👋 **Welcome!** Upload documents above to start chatting. Your data is processed securely and not stored.")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit, LangChain, and Groq AI** | Enhanced for better usability and aesthetics.")
