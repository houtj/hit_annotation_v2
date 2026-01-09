/**
 * Classification-specific UI components
 * 
 * For classification tasks:
 * - No point drawing mode
 * - No prediction overlay (mask)
 * - Shows predicted class with confidence bar
 * - Users click on a class to label the image
 */

import { 
  getFile, 
  getClasses, 
  getImageUrl, 
  saveLabels, 
  getClassificationPrediction,
  type FileDetail, 
  type Class, 
  type LabelDataItem,
  type ClassificationPrediction 
} from '../../../shared/api';
import { getUsername } from '../../../shared/state';

let currentFile: FileDetail | null = null;
let classes: Class[] = [];
let selectedClass: Class | null = null;
let currentLabel: LabelDataItem | null = null;  // Single class label for classification
let canvas: HTMLCanvasElement | null = null;
let ctx: CanvasRenderingContext2D | null = null;
let img: HTMLImageElement | null = null;
let prediction: ClassificationPrediction | null = null;

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Render the classification labeling UI
 */
export function renderClassificationHTML(file: FileDetail, classList: Class[]): string {
  currentFile = file;
  classes = classList;
  
  // Load existing class label
  if (file.label?.label_data) {
    const classLabel = file.label.label_data.find(item => item.type === 'class');
    if (classLabel) {
      currentLabel = classLabel;
      selectedClass = classList.find(c => c.classname === classLabel.classname) || null;
    }
  }
  
  return `
    <div class="labeling-container classification-mode">
      <div class="labeling-header">
        <button id="back-btn" class="btn-back">← Back</button>
        <h2>${escapeHtml(file.filename)}</h2>
        <div class="header-spacer"></div>
      </div>
      
      <div class="labeling-body">
        <div class="left-panel">
          <div class="toolbar">
            <button id="save-btn" class="btn-primary">💾 Save</button>
            <button id="stop-btn" class="btn-secondary">⏹ Stop Training</button>
          </div>
          <div class="canvas-container">
            <canvas id="annotation-canvas"></canvas>
          </div>
          
          <!-- Prediction display (instead of overlay) -->
          <div class="prediction-display" id="prediction-display">
            <h4>Model Prediction</h4>
            <div id="prediction-content">
              <span class="no-prediction">No prediction available</span>
            </div>
          </div>
        </div>
        
        <div class="right-panel">
          <div class="classes-section">
            <h3>Select Class</h3>
            <p class="class-hint">Click a class to label this image</p>
            <div id="classes-list" class="classes-list classification-classes"></div>
          </div>
          
          <div class="current-label-section">
            <h3>Current Label</h3>
            <div id="current-label-display">
              ${currentLabel 
                ? `<span class="label-badge" style="background-color: ${selectedClass?.color || '#888'}">${escapeHtml(currentLabel.classname)}</span>` 
                : '<span class="no-label">Not labeled</span>'
              }
            </div>
          </div>
          
          <div class="progress-section">
            <h3>Training Progress</h3>
            <div class="metrics-display">
              <div class="latest-metrics">
                <span>Train: <strong id="latest-train">-</strong></span>
                <span>Test: <strong id="latest-test">-</strong></span>
              </div>
              <canvas id="metrics-chart"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

/**
 * Setup the canvas for classification (display only, no drawing)
 */
export function setupClassificationCanvas(): void {
  canvas = document.getElementById('annotation-canvas') as HTMLCanvasElement;
  if (!canvas || !currentFile) return;
  
  ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // Set canvas size to image size
  canvas.width = currentFile.width;
  canvas.height = currentFile.height;
}

/**
 * Load and display the image
 */
export async function loadClassificationImage(): Promise<void> {
  if (!canvas || !ctx || !currentFile) return;
  
  return new Promise((resolve) => {
    img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      redrawClassificationCanvas();
      resolve();
    };
    
    img.onerror = () => {
      console.error('Failed to load image');
      resolve();
    };
    
    img.src = getImageUrl(currentFile!.id);
  });
}

/**
 * Redraw the canvas (just the image for classification)
 */
function redrawClassificationCanvas(): void {
  if (!canvas || !ctx || !img) return;
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw image
  ctx.drawImage(img, 0, 0);
}

/**
 * Render the class list for selection
 */
export function renderClassificationClasses(): void {
  const classList = document.getElementById('classes-list');
  if (!classList) return;
  
  classList.innerHTML = classes.map(cls => `
    <div class="class-item ${selectedClass?.classname === cls.classname ? 'selected' : ''}"
         data-classname="${cls.classname}">
      <div class="class-color" style="background-color: ${cls.color}"></div>
      <div class="class-name">${escapeHtml(cls.classname)}</div>
    </div>
  `).join('');
}

/**
 * Handle class selection for classification
 */
export function handleClassificationClassSelect(classname: string): void {
  const cls = classes.find(c => c.classname === classname);
  if (!cls) return;
  
  selectedClass = cls;
  currentLabel = {
    type: 'class',
    classname: cls.classname,
    color: cls.color,
    origin: 'human'
  };
  
  // Update UI
  renderClassificationClasses();
  updateCurrentLabelDisplay();
}

/**
 * Update the current label display
 */
function updateCurrentLabelDisplay(): void {
  const display = document.getElementById('current-label-display');
  if (!display) return;
  
  if (currentLabel && selectedClass) {
    display.innerHTML = `<span class="label-badge" style="background-color: ${selectedClass.color}">${escapeHtml(currentLabel.classname)}</span>`;
  } else {
    display.innerHTML = '<span class="no-label">Not labeled</span>';
  }
}

/**
 * Load and display classification prediction
 */
export async function loadClassificationPrediction(fileId: number): Promise<void> {
  try {
    prediction = await getClassificationPrediction(fileId);
    updatePredictionDisplay();
  } catch (error) {
    console.error('Failed to load prediction:', error);
  }
}

/**
 * Update the prediction display
 */
function updatePredictionDisplay(): void {
  const content = document.getElementById('prediction-content');
  if (!content) return;
  
  if (!prediction) {
    content.innerHTML = '<span class="no-prediction">No prediction available</span>';
    return;
  }
  
  const confidencePercent = Math.round(prediction.confidence * 100);
  const classColor = classes.find(c => c.classname === prediction!.class)?.color || '#888';
  
  content.innerHTML = `
    <div class="prediction-result">
      <div class="predicted-class">
        <span class="class-badge" style="background-color: ${classColor}">${escapeHtml(prediction.class)}</span>
      </div>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width: ${confidencePercent}%; background-color: ${classColor}"></div>
        <span class="confidence-text">${confidencePercent}% confident</span>
      </div>
      <div class="probabilities">
        ${Object.entries(prediction.probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([cls, prob]) => {
            const pct = Math.round(prob * 100);
            const color = classes.find(c => c.classname === cls)?.color || '#888';
            return `
              <div class="prob-item">
                <span class="prob-class">${escapeHtml(cls)}</span>
                <div class="prob-bar">
                  <div class="prob-fill" style="width: ${pct}%; background-color: ${color}"></div>
                </div>
                <span class="prob-value">${pct}%</span>
              </div>
            `;
          }).join('')}
      </div>
    </div>
  `;
}

/**
 * Handle receiving a new prediction from WebSocket
 */
export function onClassificationPredictionReady(fileId: number): void {
  if (currentFile?.id === fileId) {
    loadClassificationPrediction(fileId);
  }
}

/**
 * Save the classification label
 */
export async function saveClassificationLabel(): Promise<boolean> {
  if (!currentFile || !currentLabel) {
    console.warn('No file or label to save');
    return false;
  }
  
  const username = getUsername();
  if (!username) {
    console.warn('No username set');
    return false;
  }
  
  try {
    await saveLabels(currentFile.id, [currentLabel], username);
    console.log('Classification label saved');
    return true;
  } catch (error) {
    console.error('Failed to save classification label:', error);
    return false;
  }
}

/**
 * Get current label data for saving
 */
export function getCurrentClassificationLabel(): LabelDataItem[] {
  return currentLabel ? [currentLabel] : [];
}

/**
 * Reset classification state
 */
export function resetClassificationState(): void {
  currentFile = null;
  classes = [];
  selectedClass = null;
  currentLabel = null;
  canvas = null;
  ctx = null;
  img = null;
  prediction = null;
}
