# Phase 1 Imports

import streamlit as st

# Phase 2 Imports

import os # To access Environment Variables Securely
from langchain_groq import ChatGroq # To access directly fast models using groq for langchain
from langchain_core.output_parsers import StrOutputParser # it converts raw response from model to raw string 
from langchain_core.prompts import ChatPromptTemplate # Role Separation for model , Dynamic Injection , layouts

# Phase 3 Imports

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import tempfile 


try:
    groq_api_key=st.secrets["GROQ_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

st.title("DocuMind ChatBot")
st.write("Upload a PDF and ask any question. The AI will answer based on the document's content!")

uploaded_file = st.file_uploader(
    "📃  Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False , suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

# We Will steup a session state to hold all the messages 
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Displaying All The messages    
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


@st.cache_resource
def get_vector_store(pdf_path):

    # loadPdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    docs = text_splitters.split_documents(documents)
    
    
    # create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
    ) 
    
    # Vector Store
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )
    
    return vector_store
    
if uploaded_file is not None :
    if (
        "current_pdf" not in st.session_state 
        or st.session_state.current_pdf != uploaded_file.name
    ):
        st.cache_resource.clear()

        # Delete old vector store
        if "vector_store" in st.session_state:
            del st.session_state.vector_store

        # Clear chat
        st.session_state.messages = []
        
        # with st.spinner("📚 Creating Vector Store , Please Wait ... "):
        with st.spinner("📚 Uplaoding PDF , Please Wait ... "):
            st.session_state.vector_store = get_vector_store(pdf_path)
            
        st.session_state.current_pdf = uploaded_file.name
        st.success("PDF Uploaded SuccessFully! Now You Can Ask Questions.")

prompt = st.chat_input("Pass Your prompt here please!")

if uploaded_file is None:
    st.info("Please Upload A File First.")
    st.stop()
    

if prompt:
    # Now append all the prompts in messages []
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({
        'role':'user',
        'content':prompt
    })
    

   
    
    # vector_store = get_vector_store(pdf_path)
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": 3}
        )

    groq_system_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Answer the user's question ONLY using the context below.

Context:
{context}

Question:
{user_prompt}

If the answer is not found in the context, simply say:
"I couldn't find that information in the document."
""")
    
    model = "llama-3.3-70b-versatile"   
    chat_groq = ChatGroq(
        groq_api_key = groq_api_key,
        model=model
    )
    
    
    
    chain = groq_system_prompt | chat_groq | StrOutputParser()

   

    try:
        docs = retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        response = chain.invoke({
            "context":context,
            "user_prompt":prompt
        })
        
    except Exception as e:
        st.error(f"An Error occurred: {str(e)}")
        response = "Sorry, I encountered an error processing your request."
            
    
    
    
    # response = "I am your Assistant."
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({
        'role':'assistant',
        'content':response
    })
