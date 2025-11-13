/**
 * Main entry point for the application
 */

import { isLoggedIn } from './shared/state';
import { renderWelcomeView, attachWelcomeListeners } from './pages/welcome/WelcomeView';
import { renderFilesView } from './pages/files/FilesView';
import { initializeWebSocket } from './websocket';

console.log('Active Learning Annotation Tool Starting...');

// Initialize the application
async function init() {
  try {
    // Check if user is logged in
    if (isLoggedIn()) {
      await showFilesPage();
    } else {
      showWelcomePage();
    }
    
    console.log('Application initialized successfully');
  } catch (error) {
    console.error('Failed to initialize application:', error);
  }
}

// Show welcome page
function showWelcomePage() {
  const mainContent = document.getElementById('main-content');
  if (!mainContent) return;

  mainContent.innerHTML = renderWelcomeView(onLogin);
  attachWelcomeListeners(onLogin);
}

// Show files page
async function showFilesPage() {
  // Initialize WebSocket for real-time updates
  initializeWebSocket();
  
  await renderFilesView(onLogout);
}

// Handle login
function onLogin() {
  showFilesPage();
}

// Handle logout
function onLogout() {
  showWelcomePage();
}

// Start the application
init();

