import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load PDF file under /rawdata
documents = []
for filename in os.listdir("rawdata"):
    if filename.endswith(".pdf"):
        print(f"Loading {filename}...")
        loader = PyPDFLoader(os.path.join("rawdata", filename))
        documents.extend(loader.load())
        
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("index")