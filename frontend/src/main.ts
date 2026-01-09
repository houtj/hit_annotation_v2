/**
 * Main entry point for the application
 */

import { isLoggedIn, appState } from './shared/state';
import { renderWelcomeView, attachWelcomeListeners } from './pages/welcome/WelcomeView';
import { renderFilesView } from './pages/files/FilesView';
import { renderLabelingView, cleanupLabelingView } from './pages/labeling/LabelingView';
import { initializeWebSocket } from './websocket';

console.log('Active Learning Annotation Tool Starting...');

// Initialize the application
async function init() {
  try {
    // Initialize app state (load task type from backend)
    await appState.initialize();
    
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
  
  await renderFilesView(onLogout, onFileClick);
}

// Show labeling page
async function showLabelingPage(fileId: number) {
  await renderLabelingView(fileId, onBackToFiles);
}

// Handle file click
function onFileClick(fileId: number) {
  // Clean up previous labeling view if navigating from another labeling page
  cleanupLabelingView();
  showLabelingPage(fileId);
}

// Handle back to files
function onBackToFiles() {
  // Clean up labeling view resources (especially WebSocket)
  cleanupLabelingView();
  showFilesPage();
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

