/**
 * API client for backend communication
 */

const API_BASE = '/api';

// Generic fetch wrapper with error handling
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Files API
// ============================================================================

export interface FileItem {
  id: number;
  filename: string;
  filepath: string;
  labeled: 'no' | 'manual' | 'auto';
  labeler: string;
}

export interface FileStats {
  total_files: number;
  manual_labeled: number;
  model_version: string;
}

export async function getFiles(): Promise<FileItem[]> {
  return fetchAPI<FileItem[]>('/files/');
}

export async function getFileStats(): Promise<FileStats> {
  return fetchAPI<FileStats>('/files/stats');
}

