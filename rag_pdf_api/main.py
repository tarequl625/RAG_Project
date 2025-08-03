from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.file_loader import process_pdf
from utils.embed_store import embed_chunks, query_chunks
from utils.llm_utils import ask_llm

app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    chunks = await process_pdf(file)
    embed_chunks(chunks)
    return {"message": f"{file.filename} processed and stored."}

@app.post("/query")
async def query_pdf(data: dict):
    question = data.get("question")
    related_chunks = query_chunks(question)
    answer = ask_llm(question, related_chunks)
    return JSONResponse({
        "context": related_chunks,
        "answer": answer
    })
