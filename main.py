import os
import requests
from dotenv import load_dotenv
from typing import Optional, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA

# --- Step 1: Load API Key from .env ---
load_dotenv()
GMI_API_KEY = os.getenv("GMI_API_KEY")
if not GMI_API_KEY:
    raise EnvironmentError("GMI_API_KEY is missing in .env file.")

vectorstore_path = "index"
vectorstore = FAISS.load_local(vectorstore_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Step 4: Define GMI API LLM Wrapper ---
class GmiLLM(LLM):
    model: str = "deepseek-ai/DeepSeek-V3-0324"
    temperature: float = 0.5
    max_tokens: int = 4096

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {GMI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(
            "https://api.gmi-serving.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "gmi-llm"

# --- Step 5: Build RAG Pipeline ---
llm = GmiLLM()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Step 6: Query Loop ---
while True:
    query = input("\nEnter your question (or type 'exit'): ")
    if query.lower() in {"exit", "quit"}:
        break
    answer = qa_chain.invoke(query)
    formatted_answer = answer["result"]
    print(formatted_answer)
