from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

import os

app = FastAPI()

DB_DIR = "vectorstore"
os.makedirs(DB_DIR, exist_ok=True)

# LLM
llm = OllamaLLM(model="mistral", base_url="http://127.0.0.1:11434/")
embeddings = OllamaEmbeddings(model="mistral", base_url="http://127.0.0.1:11434/")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename
        filepath = f"documents/{filename}"
        os.makedirs("documents", exist_ok=True)

        # Save file
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Load document
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Store in Chroma
        Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)

        return {"message": "Document uploaded & indexed successfully!"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/ask")
def ask(query: str = Query(...)):
    try:
        # Load vector DB
        vectordb = Chroma(embedding_function=embeddings, persist_directory=DB_DIR)

        # Retrieve relevant docs
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        Use the below context to answer the question.

        Context:
        {context}

        Question: {query}

        Answer:
        """

        answer = llm.invoke(prompt)

        return {"question": query, "answer": answer}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})