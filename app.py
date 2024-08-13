import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
import openai

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorStore(text_chunks):
    embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-x1")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorStore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def main():
    load_dotenv()  # Load environment variables from .env file
    openai.api_key = os.getenv('OPENAI_API_KEY')  # Set the API key for OpenAI

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html =  True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    st.header("Chat with multiple PDFs :books:")
    st.text_input("Ask a question about your documents:")

    st.write(user_template.replacee("{{MSG}}", "helllo robot"),  unsafe_allow_html = True)
    st.write(bot_template.replace("{{MSG}}", "helllo human"),  unsafe_allow_html = True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    st.write("Raw text extracted from PDFs:")
                    st.write(raw_text[:2000])  # Show only the first 2000 characters for preview
                    
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.write("Text chunks:")
                    st.write(text_chunks[:5])  # Show only the first 5 chunks for preview
                    
                    # Create vector store
                    vectorStore = get_vectorStore(text_chunks)
                    st.write("Vector store created.")

                    # create onversation chain
                    st.session_state.conversation = get_conversation_chain(vectorStore)

            else:
                st.warning("Please upload at least one PDF file to process.")

if __name__ == '__main__':
    main()