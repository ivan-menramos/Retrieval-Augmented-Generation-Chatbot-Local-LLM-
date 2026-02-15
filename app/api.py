from fastapi import FastAPI
from app.rag_chain import build_rag
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Construimos el sistema RAG una sola vez al iniciar el servidor
qa_chain = build_rag()

@app.post("/ask") 
async def ask_question(question: str): 
    response = qa_chain.invoke({"query": question}) 
    return {"answer": response["result"]}
