/**
 * Application state management
 */

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

