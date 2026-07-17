# Phase 1 Imports

import streamlit as st

# Phase 2 Imports

import os # To access Environment Variables Securely
from langchain_groq import ChatGroq # To access directly fast models using groq for langchain
from langchain_core.output_parsers import StrOutputParser # it converts raw response from model to raw string 
from langchain_core.prompts import ChatPromptTemplate # Role Separation for model , Dynamic Injection , layouts

# Phase 3 Imports
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import tempfile 
import hashlib
from collections import defaultdict

# API Key
try:
    groq_api_key=st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

# Ui
st.set_page_config(layout="centered" , page_title="DocuMind ChatBot")
st.title("📃 DocuMind ChatBot")
st.write("Upload a PDF and ask any question. The AI will answer based on the document's content!")

uploaded_file = st.file_uploader(
    "📃  Upload a PDF",
    type=["pdf"]
)


# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_collection" not in st.session_state:
    st.session_state.current_collection = None    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None    
    
# Previous Messages with sources
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


@st.cache_resource
def get_vector_store(pdf_path , collection_name):

    # loadPdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks ( Better Chunking )
    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        separators=["\n\n","\n",". ","? ","! "," ",""]
    )
    
    docs = text_splitters.split_documents(documents)
    
    # Checking if document is empty or not
    if len(docs) == 0:
        st.warning("⚠️ No text could be extracted from this PDF. It might be scanned or image‑based.")
        return None
    
    # create embeddings
    embeddings = FastEmbedEmbeddings(
        model_name = "BAAI/bge-small-en"
    ) 
    
    # Vector Store
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name
    )
    
    return vector_store

    
# Handle New PDF Upload 
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False , suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
        
    unique_id = hashlib.md5(f"{uploaded_file.name}{uploaded_file.size}".encode()).hexdigest()
        
    new_collection = f"pdf_{unique_id}"
        
    if st.session_state.current_collection != new_collection:
        get_vector_store.clear()
            
        with st.spinner("📚 Indexing PDF - Please wait ..."):
            vector_store = get_vector_store(pdf_path , new_collection)
                
            
        if vector_store is None:
            st.error("❌ Failed to index the PDF. Please use a text‑based PDF or try OCR.")
            st.stop()    

        else:
            st.session_state.vector_store = vector_store
            st.session_state.current_collection = new_collection
            st.session_state.messages = []
            st.success("PDF Uploaded and indexed SuceessFully !")
           


# Chat Input

if uploaded_file is None:
    st.info("📁 Please Upload A File First.")
    st.stop()
    
if  st.session_state.vector_store is None:
    st.error("⚠ No Valid Document indexed. Please upload a text-based PDF.")   
    st.stop()
    
prompt = st.chat_input("Ask the question about the document please !")
    
    
# Prompt User Request
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({
        'role':'user',
        'content':prompt
    })
    
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7}
    )

    try:
        docs = retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        groq_system_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Answer the user's question ONLY using the context below.

Context:
{context}

Question:
{user_prompt}

If the answer is not found in the context, simply say:
"I couldn't find that information in the document."

And also you gives the response in the best format that can be any based on user's message.
""")
           
        model = "llama-3.3-70b-versatile"   
        chat_groq = ChatGroq(
        groq_api_key = groq_api_key,
        model=model,
        temperature=0
        )
        chain = groq_system_prompt | chat_groq | StrOutputParser()
        
        response = chain.invoke({
            "context":context,
            "user_prompt":prompt
        })

        
    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
        response = "Sorry, I encountered an error processing your request."
            
    
    # Responses
    st.chat_message("assistant").markdown(response)
    
    # Details ( sources )
    st.subheader("Sources")
    with st.expander(f"📖 {uploaded_file.name} "):
        page_chunks = defaultdict(list)
        
        for doc in docs:
            page = doc.metadata.get("page","N/A")

            if isinstance(page ,int):
                page+=1
                
            if doc.page_content not in page_chunks[page]:
                page_chunks[page].append(doc.page_content)    
        
        for page in sorted(page_chunks):
            st.markdown(f"📃 Page :  {page}")
            for chunk in page_chunks[page]:
                st.markdown(chunk)
                st.divider()
    
    st.session_state.messages.append({
        'role':'assistant',
        'content':response
    })
