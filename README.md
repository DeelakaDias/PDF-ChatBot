# Chat with Multiple PDFs

## Overview

This Streamlit application allows users to interact with and extract information from multiple PDF documents. By uploading PDF files, the application processes and indexes the text, enabling users to ask questions and receive responses based on the content of those documents. The application uses OpenAI’s language models to facilitate conversational queries over the indexed text.

## Features
User Interface https://github.com/user-attachments/assets/21f3811d-7efd-4a1f-851a-c0cbf0392faf 

- **Upload PDFs:** Users can upload multiple PDF documents for processing.
- **Extract Text:** The application extracts and indexes text from the uploaded PDFs.
- **Ask Questions:** Users can ask questions about the content of the PDFs.
- **Conversational Interface:** Uses OpenAI’s language models to provide conversational responses based on the indexed text.

## Requirements

Before running the application, ensure you have the following Python packages installed:

- `streamlit`
- `python-dotenv`
- `PyPDF2`
- `langchain`
- `langchain_community`
- `openai`
- `faiss-cpu` or `faiss-gpu` (depending on your system)

You can install these dependencies using pip:

```bash
pip install streamlit python-dotenv PyPDF2 langchain langchain_community openai faiss-cpu
