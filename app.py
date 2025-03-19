import os
import uuid
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
import aiofiles
from pathlib import Path
import time

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

# Performance configuration
PERFORMANCE_CONFIG = {
    "max_workers": 8,  # Increased number of worker threads for parallel processing
    "use_caching": True,  # Enable/disable caching of processed documents
    "timeout": 60.0,  # Timeout for embedding requests in seconds
    "batch_size": 20,  # Increased batch size for embedding generation
    "chunk_size": 2048,  # Larger chunks mean fewer embeddings to generate
    "use_multithreading": True,  # Enable multithreading for faster processing
    "use_fast_embeddings": True,  # Use faster embedding models when available
}

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
Settings.chunk_size = PERFORMANCE_CONFIG["chunk_size"]  # Use the size from performance config
Settings.chunk_overlap = 20

# Configure embedding batch size for better performance
Settings.embed_batch_size = PERFORMANCE_CONFIG["batch_size"]  # Process embeddings in batches

# Enable parallel processing
Settings.num_workers = PERFORMANCE_CONFIG["max_workers"]

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
    
    # Use a more efficient file writing approach for large files
    # Read and write in chunks to avoid loading the entire file into memory
    chunk_size = 1024 * 1024  # 1MB chunks
    async with aiofiles.open(file_path, 'wb') as out_file:
        # Read file in chunks
        while chunk := await file.read(chunk_size):
            await out_file.write(chunk)
    
    # Get file size to estimate processing time
    file_size = os.path.getsize(file_path)
    estimated_time = max(5, int(file_size / (1024 * 1024) * 0.5))  # Rough estimate: 0.5 seconds per MB, minimum 5 seconds
    
    # Initialize processing status with time estimate
    processing_status[document_id] = ProcessingStatus(
        status="queued",
        progress=0.0,
        message=f"Document queued for processing. Estimated time: {estimated_time} seconds"
    )
    
    # For very large files, use a separate process to avoid blocking the main thread
    if file_size > 10 * 1024 * 1024:  # If file is larger than 10MB
        print(f"Large file detected ({file_size/1024/1024:.2f}MB). Using optimized processing.")
        # Set a flag in the processing status
        processing_status[document_id] = ProcessingStatus(
            status="processing",
            progress=5.0,
            message=f"Large document detected ({file_size/1024/1024:.2f}MB). Optimizing processing..."
        )
    
    # Start background processing
    asyncio.create_task(process_document(document_id, file_path))
    
    return {"document_id": document_id}

async def process_document(document_id: str, file_path: str):
    try:
        # Check if we have a cached index for this file
        cache_path = f"processed/{document_id}.json"
        if os.path.exists(cache_path):
            try:
                # Import necessary components for loading from disk
                from llama_index.core import load_index_from_storage
                from llama_index.core.storage.storage_context import StorageContext
                
                # Load from disk
                storage_context = StorageContext.from_defaults(persist_dir=f"processed/{document_id}")
                index = load_index_from_storage(storage_context)
                
                # Save index in memory
                active_documents[document_id] = index
                
                # Update status to complete
                processing_status[document_id] = ProcessingStatus(
                    status="complete",
                    progress=100.0,
                    message="Document loaded from cache"
                )
                return
            except Exception as e:
                print(f"Error loading from cache: {e}")
                # Continue with normal processing if cache loading fails
        
        # Update status to processing
        processing_status[document_id] = ProcessingStatus(
            status="processing",
            progress=10.0,
            message="Loading document"
        )
        
        # Load document using PDFReader with optimized settings
        pdf_reader = PDFReader()
        documents = pdf_reader.load_data(file_path)
        
        # Update progress
        processing_status[document_id] = ProcessingStatus(
            status="processing",
            progress=30.0,
            message="Chunking document by paragraphs"
        )
        
        # Parse nodes with optimized chunking strategy
        parser = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        
        # Use parallel processing for chunking if enabled
        if PERFORMANCE_CONFIG["use_multithreading"]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG["max_workers"]) as executor:
                # Process documents in parallel
                future_to_doc = {executor.submit(parser.get_nodes_from_documents, [doc]): doc for doc in documents}
                all_nodes = []
                for future in concurrent.futures.as_completed(future_to_doc):
                    try:
                        doc_nodes = future.result()
                        all_nodes.extend(doc_nodes)
                    except Exception as e:
                        print(f"Error processing document chunk: {e}")
                nodes = all_nodes
        else:
            # Standard sequential processing
            nodes = parser.get_nodes_from_documents(documents)
        
        # Update progress
        processing_status[document_id] = ProcessingStatus(
            status="processing",
            progress=60.0,
            message="Generating embeddings"
        )
        
        # Import Ollama embedding model here to avoid import errors
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        # Use a faster embedding model if configured
        if PERFORMANCE_CONFIG["use_fast_embeddings"]:
            # Try to use a faster embedding model if available
            try:
                # First try to use a local embedding model for speed
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                print("Using HuggingFace embedding model for faster processing")
            except Exception as e:
                print(f"Error loading HuggingFace model: {e}")
                # Fall back to Ollama
                embed_model = OllamaEmbedding(model_name="llama3.2", request_timeout=PERFORMANCE_CONFIG["timeout"])
        else:
            # Use Ollama as configured
            embed_model = OllamaEmbedding(model_name="llama3.2", request_timeout=PERFORMANCE_CONFIG["timeout"])
        
        # Create index with explicit embedding model and optimized settings
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        
        # Save index in memory
        active_documents[document_id] = index
        
        # Save index to disk for future use
        try:
            index.storage_context.persist(persist_dir=f"processed/{document_id}")
            # Create a marker file to indicate this document has been cached
            with open(cache_path, 'w') as f:
                f.write('{}')
        except Exception as e:
            print(f"Error saving to cache: {e}")
        
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
