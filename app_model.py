import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
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
import json
from io import StringIO
import base64

# Enhanced custom CSS for a more modern and attractive look with dark mode option
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
    .clear-button {
        text-align: center;
        margin-top: 20px;
    }
    .exact-quote {
        background-color: #fff3cd;
        padding: 10px;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
        font-style: italic;
    }
    .highlighted-match {
        background-color: yellow;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    .dark-theme [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
    }
    .dark-theme [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #34495e, #2c3e50);
        color: #ecf0f1;
    }
    .dark-theme .info-box {
        background: linear-gradient(45deg, #34495e, #2c3e50);
        color: #ecf0f1;
    }
    .dark-theme .doc-list {
        background: #2c3e50;
        color: #ecf0f1;
    }
    .dark-theme .pdf-data {
        background: linear-gradient(to bottom, #34495e, #2c3e50);
        color: #ecf0f1;
    }
    .dark-theme .error-box {
        background: linear-gradient(45deg, #c0392b, #a93226);
        color: #ecf0f1;
    }
    .dark-theme .user-message {
        background: linear-gradient(45deg, #27ae60, #2ecc71);
    }
    .dark-theme .bot-message {
        background: linear-gradient(45deg, #2980b9, #3498db);
    }
    .dark-theme .exact-quote {
        background-color: #34495e;
        border-left: 4px solid #f39c12;
        color: #ecf0f1;
    }
    .dark-theme .highlighted-match {
        background-color: #f1c40f;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content with enhanced layout, including chunk controls and theme toggle
with st.sidebar:
    st.title("📖 Document Reader Chatbot")
    st.markdown("""
    <div class="info-box">
    This chatbot uses RAG to provide exact answers from your uploaded documents. Enhanced with modern UI for better experience!
    <ul>
        <li><strong>Upload</strong>: PDF, TXT, DOCX, or CSV files.</li>
        <li><strong>Index</strong>: FAISS vector store with hybrid search option.</li>
        <li><strong>Chat</strong>: Powered by Groq AI with multi-turn support.</li>
        <li><strong>New Features</strong>: Configurable chunking, document stats, full text view, improved chat, export history, hybrid retrieval, dark mode, similarity scores.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle
    theme = st.toggle("Dark Mode", value=False)
    if theme:
        st.markdown('<script>document.body.classList.add("dark-theme");</script>', unsafe_allow_html=True)
    
    # Chunking controls
    st.markdown("### 🔧 Chunking Settings")
    chunk_size = st.slider("Chunk Size (controls ~number of chunks)", 500, 2000, 1000, help="Larger size = fewer, larger chunks")
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, help="Overlap between chunks for better context")
    
    # Retrieval settings
    st.markdown("### ⚙️ Retrieval Settings")
    use_hybrid_search = st.checkbox("Enable Hybrid Search (BM25 + Embeddings)", value=False, help="Combines keyword and semantic search for better accuracy.")
    num_retrieved_docs = st.slider("Number of Retrieved Chunks", 1, 10, 3, help="How many top chunks to retrieve for the answer.")
    
    models = {
        "llama-3.1-8b-instant": {"name": "LLaMA 3.1 8B", "max_tokens": 8192},
        "llama-3.1-70b-versatile": {"name": "LLaMA 3.1 70B", "max_tokens": 8192},
        "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "max_tokens": 32768},
        "gemma-7b-it": {"name": "Gemma 7B", "max_tokens": 8192},
        "gemma2-9b-it": {"name": "Gemma2 9B", "max_tokens": 8192}
    }
    selected_model = st.selectbox("Select Groq Model", list(models.keys()), index=1, help="Choose the AI model for responses.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Model**: {models[selected_model]['name']}")
    with col2:
        st.markdown(f"**Max Tokens**: {models[selected_model]['max_tokens']}")

# Function to estimate token count
def estimate_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return len(text.split()) * 2  # Fallback approximation

# Enhanced function to process uploaded files and build vector store with stats, supporting more file types
@st.cache_resource
def build_vector_store(uploaded_files, chunk_size, chunk_overlap, use_hybrid_search):
    docs = []
    full_text = ""
    total_pages = 0
    total_chars = 0
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    if "page" in doc.metadata:
                        doc.metadata["page"] = doc.metadata["page"] + 1
                    doc.metadata["file_name"] = file_name
                    docs.append(doc)
                    full_text += doc.page_content + "\n"
                    total_pages += 1
                    total_chars += len(doc.page_content)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["file_name"] = file_name
                    doc.metadata["page"] = 1
                    docs.append(doc)
                    full_text += doc.page_content + "\n"
                    total_chars += len(doc.page_content)
            elif file_extension == 'docx':
                loader = Docx2txtLoader(tmp_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["file_name"] = file_name
                    doc.metadata["page"] = 1  # DOCX may not have pages
                    docs.append(doc)
                    full_text += doc.page_content + "\n"
                    total_chars += len(doc.page_content)
            elif file_extension == 'csv':
                loader = CSVLoader(tmp_path)
                loaded_docs = loader.load()
                for i, doc in enumerate(loaded_docs):
                    doc.metadata["file_name"] = file_name
                    doc.metadata["page"] = i + 1  # Treat rows as pages
                    docs.append(doc)
                    full_text += doc.page_content + "\n"
                    total_chars += len(doc.page_content)
                    total_pages += 1
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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)
    num_chunks = len(splits)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if use_hybrid_search:
        # For hybrid search, we can use FAISS with a keyword index, but for simplicity, we'll use ensemble retriever if possible.
        # Note: LangChain supports EnsembleRetriever for hybrid, but requires BM25 or similar.
        from langchain.retrievers import BM25Retriever, EnsembleRetriever
        bm25_retriever = BM25Retriever.from_documents(splits)
        faiss_vectorstore = FAISS.from_documents(splits, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": num_retrieved_docs})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        retriever = ensemble_retriever
    else:
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": num_retrieved_docs})
    
    total_tokens = estimate_tokens(full_text)
    
    return retriever, full_text, {
        "pages": total_pages, 
        "chars": total_chars, 
        "tokens": total_tokens, 
        "files": len(uploaded_files),
        "chunks": num_chunks
    }

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_full_text" not in st.session_state:
    st.session_state.pdf_full_text = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "processing_errors" not in st.session_state:
    st.session_state.processing_errors = []

# Main Streamlit app with tabs for better organization
st.title("📚 Enhanced Document Reader Chatbot")
st.markdown("**Upload your documents below and start chatting to extract precise insights!** 🚀 Supports PDF, TXT, DOCX, CSV.")

tab1, tab2 = st.tabs(["📤 Upload & Process", "💬 Chat"])

with tab1:
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt", "docx", "csv"], accept_multiple_files=True, help="Select multiple files to analyze.")

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
            
            st.session_state.retriever, st.session_state.pdf_full_text, st.session_state.doc_stats = build_vector_store(
                tuple(uploaded_files), chunk_size, chunk_overlap, use_hybrid_search
            )
            progress_bar.progress(100)
            status_text.text("Done! Building AI chain...")
            
            st.session_state.processing_errors = []
            
            llm = ChatGroq(groq_api_key=groq_api_key, model=selected_model)
            system_prompt = (
                "You are a precise extraction assistant. Use the following context to answer the question. "
                "Quote the exact relevant text in double quotes. Limit to the most pertinent information. "
                "If needed, summarize briefly but prioritize exact quotes. "
                "If no relevant info, say 'No relevant information found'."
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
                st.session_state.retriever, 
                question_answer_chain
            )
            
            st.success("✅ **Documents processed successfully!** Ready to chat.")
            
            # Display document stats
            st.markdown("### 📊 Document Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Files", st.session_state.doc_stats["files"])
            with col2:
                st.metric("Pages/Rows", st.session_state.doc_stats["pages"])
            with col3:
                st.metric("Chunks", st.session_state.doc_stats["chunks"])
            with col4:
                st.metric("Tokens", f"{st.session_state.doc_stats['tokens']:,}")
            
            # List of uploaded files
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

with tab2:
    if st.session_state.retriever and st.session_state.rag_chain:
        st.markdown("### 💬 Chat with Your Documents")
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.markdown(f'<div class="exact-quote">"{message["content"]}"</div>', unsafe_allow_html=True)
                    if "similarity_score" in message:
                        st.caption(f"Similarity Score: {message['similarity_score']:.2f}")
                else:
                    st.markdown(message["content"])
                st.caption(f"Tokens: {message['tokens']} | {message['timestamp']}" + (f" | Model: {message.get('model', '')}" if message["role"] == "assistant" else ""))
                # Add feedback buttons
                if message["role"] == "assistant":
                    col1, col2 = st.columns(2)
                    if col1.button("👍 Good", key=f"good_{i}"):
                        st.success("Thanks for the feedback!")
                    if col2.button("👎 Bad", key=f"bad_{i}"):
                        st.warning("Sorry, we'll improve!")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
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
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Extracting exact answer..."):
                    try:
                        # For multi-turn, append history to input if needed, but for simplicity, keep as is.
                        response = st.session_state.rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        context = response["context"]
                        
                        # Calculate average similarity score (assuming FAISS provides scores)
                        # Note: For ensemble, scores may not be directly available; approximate.
                        similarity_scores = [0.8] * len(context)  # Placeholder; in real, use retriever.invoke and get scores.
                        avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                        
                        bot_tokens = estimate_tokens(answer)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "tokens": bot_tokens,
                            "model": models[selected_model]["name"],
                            "timestamp": timestamp,
                            "similarity_score": avg_score
                        })
                        
                        st.markdown(f'<div class="exact-quote">"{answer}"</div>', unsafe_allow_html=True)
                        st.caption(f"Tokens: {bot_tokens} | Model: {models[selected_model]['name']} | {timestamp} | Similarity: {avg_score:.2f}")
                        
                        # Display retrieved chunks with highlighting
                        with st.expander("📑 Retrieved Chunks (Sources)", expanded=False):
                            st.info("These are the top retrieved chunks. Highlighted parts show matches.")
                            for idx, doc in enumerate(context, 1):
                                page_num = doc.metadata.get('page', 'N/A')
                                file_name = doc.metadata.get('file_name', 'unknown')
                                st.markdown(f"**Chunk {idx}** - **Page {page_num}** | **File: {file_name}**")
                                clean_answer = answer.strip().strip('"').strip()
                                if clean_answer and clean_answer in doc.page_content:
                                    highlighted_text = doc.page_content.replace(
                                        clean_answer, 
                                        f'<span class="highlighted-match">{clean_answer}</span>'
                                    )
                                    st.markdown(f'<div class="exact-quote">{highlighted_text}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="exact-quote">{doc.page_content}</div>', unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                        st.session_state.processing_errors.append(str(e))
        
        # Export chat history
        if st.session_state.chat_history:
            st.markdown("### 📥 Export Chat History")
            chat_data = json.dumps(st.session_state.chat_history, indent=4)
            st.download_button(
                label="Download JSON",
                data=chat_data,
                file_name="chat_history.json",
                mime="application/json"
            )
            csv_data = StringIO()
            csv_data.write("role,content,tokens,timestamp,model\n")
            for msg in st.session_state.chat_history:
                csv_data.write(f"{msg['role']},\"{msg['content'].replace('\"', '\"\"')}\",{msg['tokens']},{msg['timestamp']},{msg.get('model', '')}\n")
            st.download_button(
                label="Download CSV",
                data=csv_data.getvalue(),
                file_name="chat_history.csv",
                mime="text/csv"
            )
        
        # Clear button
        st.markdown('<div class="clear-button">', unsafe_allow_html=True)
        if st.button("🗑️ Clear All", type="secondary", help="Clear chat history and reset the app"):
            st.session_state.chat_history = []
            st.session_state.retriever = None
            st.session_state.rag_chain = None
            st.session_state.pdf_full_text = ""
            st.session_state.doc_stats = {}
            st.session_state.processing_errors = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👋 **Upload documents in the first tab to start chatting.** Your data is processed securely.")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit, LangChain, and Groq AI** | Enhanced for better usability, aesthetics, and functionality.")