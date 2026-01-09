/**
 * Application state management
 */

// Application state
class AppState {
  taskType: string | null = null;
  
  async initialize(): Promise<void> {
    // Fetch task type from backend
    try {
      const response = await fetch('/api/config');
      if (response.ok) {
        const config = await response.json();
        this.taskType = config.task;
        console.log(`Initialized app with task type: ${this.taskType}`);
      } else {
        console.error('Failed to load config from backend');
        this.taskType = 'segmentation'; // Default fallback
      }
    } catch (error) {
      console.error('Error loading config:', error);
      this.taskType = 'segmentation'; // Default fallback
    }
  }
}

export const appState = new AppState();

// Get current username from localStorage
export function getUsername(): string | null {
  return localStorage.getItem('username');
}

// Set username in localStorage
export function setUsername(username: string): void {
  localStorage.setItem('username', username);
}

// Clear username (logout)
export function clearUsername(): void {
  localStorage.removeItem('username');
}

// Check if user is logged in
export function isLoggedIn(): boolean {
  return getUsername() !== null;
}

