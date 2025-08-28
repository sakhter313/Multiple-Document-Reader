import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

# Inject custom CSS for background image and styling
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url('https://source.unsplash.com/random/1920x1080/?abstract,technology,ai');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(240, 244, 248, 0.85);
        backdrop-filter: blur(10px);
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid #cccccc;
        border-radius: 5px;
    }
    .stSpinner > div {
        color: #4CAF50;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #DCF8C6;
    }
    .ai-message {
        background-color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.title("📖 Document Chatbot")
    st.markdown("""
    Welcome to the AI-Powered Document Chatbot!
    
    - Upload multiple PDF or TXT files.
    - The app uses FAISS for vector storage and Groq AI for responses.
    - Enjoy a conversational experience with your documents.
    
    **Features:**
    - Maintains chat history.
    - Displays source context.
    - Select from latest Groq models.
    """)
    # Model selection
    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct",
        "qwen/qwen3-32b",
        "compound-beta",
        "compound-beta-mini"
    ]
    selected_model = st.selectbox("Select Groq Model", models, index=1)  # Default to llama-3.3-70b-versatile

# Function to process uploaded files and build FAISS vector store
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
                loader = PyPDFLoader(tmp_path)
            elif file_extension == 'txt':
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                continue
            docs.extend(loader.load())
        finally:
            os.unlink(tmp_path)
    
    if not docs:
        raise ValueError("No valid documents uploaded.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# Main Streamlit app
st.title("🤖 Chat with Your Documents")
st.markdown("""
Upload your documents (PDF or TXT) and start a conversation about their content.  
Powered by FAISS for retrieval and Groq for generation.
""")

# File uploader
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

# API Key from secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it in your app settings.")
else:
    groq_api_key = st.secrets["GROQ_API_KEY"]

    if uploaded_files:
        try:
            with st.spinner("Processing documents and building index... This may take a moment."):
                vectorstore = build_vector_store(tuple(uploaded_files))
            
            st.success("Index built successfully! Start chatting below.")
            
            # Set up LLM with selected model
            llm = ChatGroq(groq_api_key=groq_api_key, model=selected_model)
            
            # Prompt template for conversational chain
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Keep the answer concise."
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ]
            )
            
            # Conversational Retrieval Chain with explicit input keys
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                return_source_documents=True,
                input_key="question",  # Explicitly define input key
                output_key="answer"   # Explicitly define output key
            )
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(f'<div class="chat-message {message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)
                    if "sources" in message:
                        with st.expander("🔍 Sources"):
                            for i, doc in enumerate(message["sources"], 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.write(doc.page_content)
                                st.write(f"*Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 'N/A')}*")
                                st.divider()
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the documents:"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(f'<div class="chat-message user-message">{prompt}</div>', unsafe_allow_html=True)
                
                with st.spinner("Thinking..."):
                    try:
                        # Invoke chain with explicit input structure
                        response = qa_chain.invoke({
                            "question": prompt,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = response.get("answer", "Sorry, I couldn't generate a response.")
                        sources = response.get("source_documents", [])
                        
                        # Update chat history
                        st.session_state.chat_history.extend([
                            HumanMessage(content=prompt),
                            AIMessage(content=answer)
                        ])
                        
                        # Add AI message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        with st.chat_message("assistant"):
                            st.markdown(f'<div class="chat-message ai-message">{answer}</div>', unsafe_allow_html=True)
                            if sources:
                                with st.expander("🔍 Sources"):
                                    for i, doc in enumerate(sources, 1):
                                        st.markdown(f"**Chunk {i}:**")
                                        st.write(doc.page_content)
                                        st.write(f"*Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 'N/A')}*")
                                        st.divider()
                    except Exception as e:
                        st.error(f"Error during query: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
    else:
        st.info("Please upload at least one document to get started.")