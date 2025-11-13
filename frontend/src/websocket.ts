// WebSocket connection management

let ws: WebSocket | null = null;

export function initializeWebSocket() {
  const wsUrl = `ws://localhost:8000/ws`;
  
  console.log('Connecting to WebSocket...');
  ws = new WebSocket(wsUrl);
  
  ws.onopen = () => {
    console.log('WebSocket connected');
    updateStatus('Connected');
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('WebSocket message received:', data);
    handleWebSocketMessage(data);
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    updateStatus('Connection error');
  };
  
  ws.onclose = () => {
    console.log('WebSocket disconnected');
    updateStatus('Disconnected');
    
    // Attempt to reconnect after 3 seconds
    setTimeout(() => {
      console.log('Attempting to reconnect...');
      initializeWebSocket();
    }, 3000);
  };
}

function handleWebSocketMessage(data: any) {
  // Handle different message types
  switch (data.type) {
    case 'training_update':
      console.log('Training update:', data);
      break;
    case 'inference_complete':
      console.log('Inference complete:', data);
      break;
    case 'status':
      updateStatus(data.message);
      break;
    default:
      console.log('Unknown message type:', data);
  }
}

function updateStatus(message: string) {
  const statusElement = document.getElementById('status');
  if (statusElement) {
    statusElement.textContent = message;
  }
}

export function sendWebSocketMessage(data: any) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  } else {
    console.error('WebSocket is not connected');
  }
}

