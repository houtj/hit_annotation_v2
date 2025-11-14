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

// ============================================================================
// Labels API
// ============================================================================

export interface LabelDataItem {
  type: string;
  classname: string;
  color: string;
  x?: number;
  y?: number;
  path?: string;
}

export interface Label {
  id: number;
  label_data: LabelDataItem[];
  created_by: string;
  updated_at: string;
}

export interface FileDetail {
  id: number;
  filename: string;
  filepath: string;
  width: number;
  height: number;
  label: Label | null;
}

export interface Class {
  classname: string;
  color: string;
}

export async function getFile(fileId: number): Promise<FileDetail> {
  return fetchAPI<FileDetail>(`/files/${fileId}`);
}

export async function getClasses(): Promise<Class[]> {
  return fetchAPI<Class[]>('/classes');
}

export function getImageUrl(fileId: number): string {
  return `${API_BASE}/files/${fileId}/image`;
}

export async function saveLabels(
  fileId: number,
  labelData: LabelDataItem[],
  createdBy: string
): Promise<Label> {
  return fetchAPI<Label>(`/files/${fileId}/labels`, {
    method: 'POST',
    body: JSON.stringify({
      label_data: labelData,
      created_by: createdBy,
    }),
  });
}
