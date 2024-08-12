import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

openai_api_keys = os.getenv('OPENAI_API_KEY','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')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure that the text is not None
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorStore(text_chunks):
    embeddings =  OpenAIEmbeddings()
    vectorStore =  FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorStore

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask a question about your documents:")

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
                    
                    #create vector store
                    vectorStore = get_vectorStore(text_chunks)

            else:
                st.warning("Please upload at least one PDF file to process.")

if __name__ == '__main__':
    main()

print("commit")