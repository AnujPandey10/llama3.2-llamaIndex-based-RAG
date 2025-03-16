# llama3.2 based RAG - Fast Local Offline RAG Application

THis RAG is a fast, local, offline RAG (Retrieval Augmented Generation) application that uses Ollama's Llama3.2 and LlamaIndex. It allows you to quickly process uploaded PDFs and query document content, as well as chat with the AI model even when no document is uploaded.

## Features

- **PDF Processing**: Upload and process PDF documents by chunking them into paragraphs and generating embeddings
- **RAG Querying**: Ask questions about your documents and get answers based on their content
- **Chat Mode**: Chat with the AI model even without uploading documents
- **Local & Offline**: Runs completely offline on your local network
- **Real-time Status Updates**: See processing progress with a status bar
- **Modern UI**: Clean, responsive interface similar to ChatGPT

## Requirements

- Python 3.9+
- Ollama with Llama3.2 model installed locally

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd rag_windsor
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Make sure Ollama is running with Llama3.2 model:
```
ollama run llama3.2
```

## Usage

1. Start the application:
```
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. To access from other devices on your local network, use your computer's IP address:
```
http://<your-ip-address>:8000
```

## How to Use

1. **Upload a Document**:
   - Click the "Upload" button in the sidebar
   - Select a PDF file from your computer
   - Wait for processing to complete (progress will be shown)

2. **Chat with the AI**:
   - Type your message in the input box at the bottom
   - Toggle "Use RAG for responses" to switch between regular chat and document-based answers
   - Press Enter or click "Send" to get a response

3. **View Documents**:
   - All uploaded documents appear in the sidebar
   - The most recently processed document is used for RAG queries

## Future Enhancements

- Support for DOC, PPT, and Excel files
- Persistent storage for documents and embeddings
- User authentication
- Enhanced conversation memory and context handling

## Troubleshooting

- If you encounter issues with Ollama, make sure it's running in a separate terminal window
- For large PDFs, processing may take some time - the progress bar will keep you updated
- If the application fails to start, check that all dependencies are installed correctly

## License

[MIT License](LICENSE)
