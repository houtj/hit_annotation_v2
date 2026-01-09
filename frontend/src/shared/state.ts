/**
 * Application state management
 */

// ============================================================================
// App Configuration (loaded once at startup)
// ============================================================================

export type TaskType = 'segmentation' | 'classification';

export interface AppConfig {
  task: TaskType;
  classes: Array<{ classname: string; color: string }>;
}

// Cached app config (loaded on app init)
let appConfig: AppConfig | null = null;

/**
 * Initialize app by loading config from backend
 * Should be called once at app startup
 */
export async function initializeApp(): Promise<AppConfig> {
  if (appConfig !== null) {
    return appConfig;
  }

  try {
    const response = await fetch('/api/config');
    if (!response.ok) {
      throw new Error('Failed to load config');
    }
    appConfig = await response.json();
    console.log('[App] Initialized with task:', appConfig!.task);
    return appConfig!;
  } catch (error) {
    console.error('[App] Failed to initialize:', error);
    // Default to segmentation if config loading fails
    appConfig = {
      task: 'segmentation',
      classes: []
    };
    return appConfig;
  }
}

/**
 * Get the current task type (cached)
 */
export function getTaskType(): TaskType {
  return appConfig?.task ?? 'segmentation';
}

/**
 * Get cached app config
 */
export function getAppConfig(): AppConfig | null {
  return appConfig;
}

/**
 * Check if app has been initialized
 */
export function isAppInitialized(): boolean {
  return appConfig !== null;
}

// ============================================================================
// User Session
// ============================================================================

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
