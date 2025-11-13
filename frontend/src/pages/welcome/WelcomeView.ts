/**
 * Welcome page view - user enters their name
 */

import { setUsername } from '../../shared/state';

export function renderWelcomeView(_onLogin: () => void): string {
  return `
    <div class="welcome-container">
      <div class="welcome-card">
        <h1>Welcome to Active Learning Annotation Tool</h1>
        <p>Please enter your name to continue</p>
        <form id="welcome-form">
          <input 
            type="text" 
            id="username-input" 
            placeholder="Your name" 
            required 
            minlength="2"
            maxlength="50"
            autocomplete="off"
          />
          <button type="submit">Continue</button>
        </form>
      </div>
    </div>
  `;
}

export function attachWelcomeListeners(onLogin: () => void): void {
  const form = document.getElementById('welcome-form') as HTMLFormElement;
  const input = document.getElementById('username-input') as HTMLInputElement;

  if (form && input) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const username = input.value.trim();
      
      if (username.length >= 2) {
        setUsername(username);
        onLogin();
      }
    });

    // Focus input on load
    input.focus();
  }
}

