# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import load_rag_chain

app = FastAPI()
qa_chain = load_rag_chain()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        result = qa_chain.invoke(request.question)
        return {"question": request.question, "answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
