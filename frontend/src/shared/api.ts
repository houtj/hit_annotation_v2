/**
 * API client for backend communication
 */

const API_BASE = '/api';

// Generic fetch wrapper with error handling
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    cache: 'no-store',  // Disable caching to ensure fresh data
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
  // Add timestamp to prevent caching
  const timestamp = new Date().getTime();
  return fetchAPI<FileItem[]>(`/files/?_t=${timestamp}`);
}

export async function getFileStats(): Promise<FileStats> {
  // Add timestamp to prevent caching
  const timestamp = new Date().getTime();
  return fetchAPI<FileStats>(`/files/stats?_t=${timestamp}`);
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
  // Add timestamp to prevent caching
  const timestamp = new Date().getTime();
  return fetchAPI<FileDetail>(`/files/${fileId}?_t=${timestamp}`);
}

export async function getClasses(): Promise<Class[]> {
  return fetchAPI<Class[]>('/classes');
}

export function getImageUrl(fileId: number): string {
  // Add timestamp to prevent caching
  const timestamp = new Date().getTime();
  return `${API_BASE}/files/${fileId}/image?_t=${timestamp}`;
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

// ============================================================================
// Training Metrics API
// ============================================================================

export interface MetricEntry {
  epoch: number;
  train_loss: number;
  test_loss: number;
}

export async function getTrainingMetrics(majorVersion: number): Promise<MetricEntry[]> {
  const timestamp = new Date().getTime();
  return fetchAPI<MetricEntry[]>(`/training/metrics/${majorVersion}?_t=${timestamp}`);
}

export async function getCurrentVersion(): Promise<{version: string}> {
  const timestamp = new Date().getTime();
  return fetchAPI<{version: string}>(`/training/version?_t=${timestamp}`);
}
