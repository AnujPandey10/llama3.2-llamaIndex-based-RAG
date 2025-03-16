import os
import uuid
import asyncio
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
import aiofiles
from pathlib import Path

# Import only what we need from llama-index-core
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    use_rag: bool = False

class ProcessingStatus(BaseModel):
    status: str
    progress: float
    message: str

# In-memory storage
active_documents = {}  # document_id -> index
processing_status = {}  # document_id -> ProcessingStatus
chat_history = {}  # session_id -> List[ChatMessage]

# Configure LlamaIndex settings
Settings.chunk_size = 1024
Settings.chunk_overlap = 20

# We'll configure the LLM and embedding model when needed
# This avoids import errors with the current package structure

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_status_update(self, client_id: str, status: ProcessingStatus):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(status.dict())

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported at this time")
    
    # Generate unique ID for this document
    document_id = str(uuid.uuid4())
    
    # Save the file
    file_path = f"uploads/{document_id}.pdf"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Initialize processing status
    processing_status[document_id] = ProcessingStatus(
        status="queued",
        progress=0.0,
        message="Document queued for processing"
    )
    
    # Start background processing
    asyncio.create_task(process_document(document_id, file_path))
    
    return {"document_id": document_id}

async def process_document(document_id: str, file_path: str):
    try:
        # Update status to processing
        processing_status[document_id] = ProcessingStatus(
            status="processing",
            progress=10.0,
            message="Loading document"
        )
        
        # Load document using PDFReader
        pdf_reader = PDFReader()
        documents = pdf_reader.load_data(file_path)
        
        # Update progress
        processing_status[document_id] = ProcessingStatus(
            status="processing",
            progress=30.0,
            message="Chunking document by paragraphs"
        )
        
        # Parse nodes (chunking by paragraphs)
        parser = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        nodes = parser.get_nodes_from_documents(documents)
        
        # Update progress
        processing_status[document_id] = ProcessingStatus(
            status="processing",
            progress=60.0,
            message="Generating embeddings"
        )
        
        # Import Ollama embedding model here to avoid import errors
        from llama_index.embeddings.ollama import OllamaEmbedding
        embed_model = OllamaEmbedding(model_name="llama3.2")
        
        # Create index with explicit embedding model
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        
        # Save index in memory
        active_documents[document_id] = index
        
        # Update status to complete
        processing_status[document_id] = ProcessingStatus(
            status="complete",
            progress=100.0,
            message="Document processing complete"
        )
        
    except Exception as e:
        # Update status to error
        processing_status[document_id] = ProcessingStatus(
            status="error",
            progress=0.0,
            message=f"Error processing document: {str(e)}"
        )

@app.get("/status/{document_id}")
async def get_status(document_id: str):
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return processing_status[document_id]

@app.post("/chat")
async def chat(request: ChatRequest):
    # Initialize session if needed
    session_id = "default"  # In a real app, you'd use a proper session ID
    
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    # Add user message to history
    chat_history[session_id].extend(request.messages)
    
    # Process the message
    if request.use_rag and active_documents:
        # Import Ollama LLM here to avoid import errors
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        # Initialize Ollama models
        llm = Ollama(model="llama3.2", request_timeout=120.0)
        embed_model = OllamaEmbedding(model_name="llama3.2")
        
        # Use the most recently processed document for RAG
        document_id = list(active_documents.keys())[-1]
        index = active_documents[document_id]
        
        # Create query engine with explicit LLM
        query_engine = index.as_query_engine(llm=llm)
        
        # Get the latest user message
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
        
        try:
            # Query the document
            response = query_engine.query(user_message)
            
            # Create assistant response
            assistant_message = ChatMessage(
                role="assistant",
                content=str(response)
            )
        except Exception as e:
            # Handle any errors during querying
            assistant_message = ChatMessage(
                role="assistant",
                content=f"I encountered an error while processing your request: {str(e)}"
            )
    else:
        # Import Ollama LLM here to avoid import errors
        from llama_index.llms.ollama import Ollama
        llm = Ollama(model="llama3.2", request_timeout=120.0)
        
        # Get the latest user message
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
        
        # Get response from LLM
        response = llm.complete(user_message)
        
        # Create assistant response
        assistant_message = ChatMessage(
            role="assistant",
            content=response.text
        )
    
    # Add assistant response to history
    chat_history[session_id].append(assistant_message)
    
    return {"response": assistant_message}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Wait for messages (ping/pong or other control messages)
            data = await websocket.receive_text()
            # Process messages if needed
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/documents")
async def get_documents():
    return {
        "documents": [
            {"id": doc_id, "status": processing_status[doc_id].status}
            for doc_id in active_documents.keys()
        ]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
