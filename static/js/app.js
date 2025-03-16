document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const processingStatus = document.getElementById('processing-status');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const documentList = document.getElementById('document-list');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const ragToggle = document.getElementById('rag-toggle');
    
    // Variables
    let activeDocumentId = null;
    let websocket = null;
    const clientId = generateClientId();
    
    // Initialize WebSocket connection
    function initWebSocket() {
        websocket = new WebSocket(`ws://${window.location.host}/ws/${clientId}`);
        
        websocket.onopen = () => {
            console.log('WebSocket connection established');
        };
        
        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateProcessingStatus(data);
        };
        
        websocket.onclose = () => {
            console.log('WebSocket connection closed');
            // Try to reconnect after a delay
            setTimeout(initWebSocket, 3000);
        };
        
        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    // Initialize
    initWebSocket();
    loadDocuments();
    
    // Event Listeners
    uploadBtn.addEventListener('click', uploadDocument);
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Functions
    function generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    async function uploadDocument() {
        if (!fileInput.files || fileInput.files.length === 0) {
            alert('Please select a file to upload');
            return;
        }
        
        const file = fileInput.files[0];
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            alert('Only PDF files are supported at this time');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            // Show processing status
            processingStatus.classList.remove('hidden');
            progressBar.style.width = '0%';
            statusMessage.textContent = 'Uploading document...';
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to upload document');
            }
            
            const data = await response.json();
            activeDocumentId = data.document_id;
            
            // Start polling for status updates
            pollProcessingStatus(activeDocumentId);
            
        } catch (error) {
            console.error('Error uploading document:', error);
            statusMessage.textContent = 'Error: ' + error.message;
        }
    }
    
    async function pollProcessingStatus(documentId) {
        try {
            const response = await fetch(`/status/${documentId}`);
            if (!response.ok) {
                throw new Error('Failed to get processing status');
            }
            
            const data = await response.json();
            updateProcessingStatus(data);
            
            if (data.status === 'processing' || data.status === 'queued') {
                // Continue polling
                setTimeout(() => pollProcessingStatus(documentId), 1000);
            } else if (data.status === 'complete') {
                // Refresh document list
                loadDocuments();
                // Add system message
                addSystemMessage('Document processing complete! You can now ask questions about the document.');
            }
            
        } catch (error) {
            console.error('Error polling status:', error);
            statusMessage.textContent = 'Error checking status: ' + error.message;
        }
    }
    
    function updateProcessingStatus(data) {
        progressBar.style.width = `${data.progress}%`;
        statusMessage.textContent = data.message;
        
        if (data.status === 'complete') {
            // Change the status display to show completion
            setTimeout(() => {
                processingStatus.classList.add('hidden');
                fileInput.value = '';
            }, 2000);
        } else if (data.status === 'error') {
            // Show error state
            progressBar.style.backgroundColor = 'var(--error-color)';
        }
    }
    
    async function loadDocuments() {
        try {
            const response = await fetch('/documents');
            if (!response.ok) {
                throw new Error('Failed to load documents');
            }
            
            const data = await response.json();
            
            if (data.documents && data.documents.length > 0) {
                documentList.innerHTML = '';
                
                data.documents.forEach(doc => {
                    const li = document.createElement('li');
                    li.textContent = `Document ${doc.id.substring(0, 8)}`;
                    
                    if (doc.status === 'complete') {
                        li.style.borderColor = 'var(--success-color)';
                    } else if (doc.status === 'error') {
                        li.style.borderColor = 'var(--error-color)';
                    }
                    
                    documentList.appendChild(li);
                });
                
                // Enable RAG toggle if there's at least one complete document
                const hasCompleteDoc = data.documents.some(doc => doc.status === 'complete');
                ragToggle.disabled = !hasCompleteDoc;
                
                if (!hasCompleteDoc) {
                    ragToggle.checked = false;
                }
                
            } else {
                documentList.innerHTML = '<li class="empty-message">No documents uploaded</li>';
                ragToggle.disabled = true;
                ragToggle.checked = false;
            }
            
        } catch (error) {
            console.error('Error loading documents:', error);
            documentList.innerHTML = '<li class="empty-message">Error loading documents</li>';
        }
    }
    
    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Split content by paragraphs and create p elements
        const paragraphs = content.split('\n').filter(p => p.trim() !== '');
        paragraphs.forEach(paragraph => {
            const p = document.createElement('p');
            p.textContent = paragraph;
            messageContent.appendChild(p);
        });
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addSystemMessage(content) {
        addMessage('system', content);
    }
    
    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage('user', message);
        
        // Clear input
        chatInput.value = '';
        
        // Disable send button while processing
        sendBtn.disabled = true;
        
        try {
            const useRag = ragToggle.checked;
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: [{ role: 'user', content: message }],
                    use_rag: useRag
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to get response');
            }
            
            const data = await response.json();
            
            // Add assistant message to chat
            addMessage('assistant', data.response.content);
            
        } catch (error) {
            console.error('Error sending message:', error);
            addSystemMessage('Error: Failed to get response. Please try again.');
        } finally {
            // Re-enable send button
            sendBtn.disabled = false;
        }
    }
});
