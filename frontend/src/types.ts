/**
 * TypeScript types for Active Learning Annotation Tool
 * 
 * These types correspond to the Python SQLAlchemy models in backend/db/models.py
 */

// ============================================================================
// Database Models
// ============================================================================

/**
 * File record - represents an uploaded image
 */
export interface File {
  id: number;
  filename: string;
  filepath: string;
}

/**
 * Point label - represents a single point annotation
 */
export interface PointLabel {
  type: 'point';
  classname: string;
  color: string;
  x: number;
  y: number;
}

/**
 * Mask label - represents a segmentation mask annotation
 */
export interface MaskLabel {
  type: 'mask';
  classname: string;
  color: string;
  path: string;
}

/**
 * Union type for all label types
 */
export type LabelData = PointLabel | MaskLabel;

/**
 * Label record - stores annotations for a file
 */
export interface Label {
  id: number;
  file_id: number;
  created_by: string;
  updated_at: string; // ISO 8601 timestamp
  label_data: LabelData[];
}

/**
 * Prediction record - ML model prediction for a file
 */
export interface Prediction {
  id: number;
  file_id: number;
  path: string;
}

/**
 * Model version record - tracks ML model training
 */
export interface ModelVersion {
  version: string;
  training_start_at: string; // ISO 8601 timestamp
  training_end_at: string | null; // ISO 8601 timestamp or null
  status: 'training' | 'completed' | 'failed';
  metrics: Record<string, number> | null; // e.g., { loss: 0.123, accuracy: 0.95 }
  path: string | null;
}

/**
 * Class definition - annotation class
 */
export interface Class {
  classname: string;
  color: string; // Hex color code: #RRGGBB
}


// ============================================================================
// API Request/Response Types
// ============================================================================

/**
 * Request to create a new label
 */
export interface CreateLabelRequest {
  file_id: number;
  created_by: string;
  label_data: LabelData[];
}

/**
 * Request to update an existing label
 */
export interface UpdateLabelRequest {
  label_data: LabelData[];
}

/**
 * File with associated labels and predictions
 */
export interface FileWithAnnotations extends File {
  labels?: Label;
  predictions?: Prediction[];
}

