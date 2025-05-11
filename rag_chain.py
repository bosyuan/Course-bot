# rag_chain.py
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from model import GmiLLM

def load_rag_chain(vectorstore_path: str = "index") -> RetrievalQA:
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local(vectorstore_path, embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = GmiLLM()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
