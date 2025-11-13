// Main entry point for the application
import { initializeWebSocket } from './websocket';
import { renderUI } from './ui';

console.log('Active Learning Annotation Tool Starting...');

// Initialize the application
async function init() {
  try {
    // Render initial UI
    renderUI();
    
    // Connect to WebSocket for real-time updates
    initializeWebSocket();
    
    console.log('Application initialized successfully');
  } catch (error) {
    console.error('Failed to initialize application:', error);
  }
}

// Start the application
init();

