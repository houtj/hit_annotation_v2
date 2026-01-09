/**
 * Segmentation-specific UI components
 * 
 * For segmentation tasks:
 * - Point drawing mode for labeling
 * - Prediction overlay with opacity control
 * - Point extraction from predictions
 */

import { 
  getFile, 
  getClasses, 
  getImageUrl, 
  saveLabels, 
  getPredictionUrl,
  type FileDetail, 
  type Class, 
  type LabelDataItem 
} from '../../../shared/api';
import { getUsername } from '../../../shared/state';

let currentFile: FileDetail | null = null;
let classes: Class[] = [];
let selectedClass: Class | null = null;
let points: LabelDataItem[] = [];
let isPointMode = false;
let canvas: HTMLCanvasElement | null = null;
let ctx: CanvasRenderingContext2D | null = null;
let img: HTMLImageElement | null = null;
let predictionMask: HTMLImageElement | null = null;
let predictionOpacity: number = 0.5;
let showPoints: boolean = true;
let showPrediction: boolean = true;

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Render the segmentation labeling UI
 */
export function renderSegmentationHTML(file: FileDetail, classList: Class[]): string {
  currentFile = file;
  classes = classList;
  
  // Set first class as default
  if (classList.length > 0) {
    selectedClass = classList[0];
  }
  
  // Load existing labels
  if (file.label) {
    points = file.label.label_data.filter(item => item.type === 'point');
  } else {
    points = [];
  }
  
  return `
    <div class="labeling-container segmentation-mode">
      <div class="labeling-header">
        <button id="back-btn" class="btn-back">← Back</button>
        <h2>${escapeHtml(file.filename)}</h2>
        <div class="header-spacer"></div>
      </div>
      
      <div class="labeling-body">
        <div class="left-panel">
          <div class="toolbar">
            <button id="point-mode-btn" class="tool-btn">📍 Point Mode</button>
            <button id="extract-points-btn" class="tool-btn">🎯 Extract Points</button>
            <button id="save-btn" class="btn-primary">💾 Save</button>
            <button id="stop-btn" class="btn-secondary">⏹ Stop Training</button>
            <span id="point-count" class="point-count">Points: ${points.length}</span>
          </div>
          <div class="canvas-container">
            <canvas id="annotation-canvas"></canvas>
          </div>
        </div>
        
        <div class="right-panel">
          <div class="classes-section">
            <h3>Classes</h3>
            <div id="classes-list" class="classes-list"></div>
          </div>
          
          <div class="visibility-section">
            <h3>Display Options</h3>
            <div class="toggle-controls">
              <button id="toggle-points-btn" class="toggle-btn active">
                <span class="toggle-icon">👁️</span> Show Points
              </button>
              <button id="toggle-prediction-btn" class="toggle-btn active">
                <span class="toggle-icon">👁️</span> Show Prediction
              </button>
            </div>
          </div>
          
          <div class="prediction-section">
            <h3>Prediction Overlay</h3>
            <div class="opacity-control">
              <label for="opacity-slider">Opacity: <span id="opacity-value">50%</span></label>
              <input type="range" id="opacity-slider" min="0" max="100" value="50" />
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
 * Setup the canvas for segmentation
 */
export function setupSegmentationCanvas(): void {
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
export async function loadSegmentationImage(): Promise<void> {
  if (!canvas || !ctx || !currentFile) return;
  
  return new Promise((resolve) => {
    img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      redrawSegmentationCanvas();
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
 * Redraw the segmentation canvas with image, prediction overlay, and points
 */
export function redrawSegmentationCanvas(): void {
  if (!canvas || !ctx || !img) return;
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw image (layer 1: bottom)
  ctx.drawImage(img, 0, 0);
  
  // Draw prediction mask if available and visible (layer 2: above image, below points)
  if (predictionMask && showPrediction) {
    drawGreenPredictionMask();
  }
  
  // Draw points if visible (layer 3: on top)
  if (showPoints) {
    points.forEach(point => {
      if (point.x !== undefined && point.y !== undefined) {
        drawPoint(point.x, point.y, point.color);
      }
    });
  }
  
  // Update point count
  updatePointCount();
}

function drawGreenPredictionMask(): void {
  if (!canvas || !ctx || !predictionMask) return;
  
  // Create an offscreen canvas to process the mask
  const offscreen = document.createElement('canvas');
  offscreen.width = canvas.width;
  offscreen.height = canvas.height;
  const offscreenCtx = offscreen.getContext('2d');
  if (!offscreenCtx) return;
  
  // Draw the grayscale mask to offscreen canvas
  offscreenCtx.drawImage(predictionMask, 0, 0, canvas.width, canvas.height);
  
  // Get image data
  const imageData = offscreenCtx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  // Convert grayscale to red-green gradient
  for (let i = 0; i < data.length; i += 4) {
    const gray = data[i];
    const intensity = gray / 255.0;
    
    let r, g, b;
    
    if (intensity < 0.5) {
      r = 255;
      g = Math.floor(intensity * 2 * 255);
      b = Math.floor(intensity * 2 * 255);
    } else {
      r = Math.floor((1 - intensity) * 2 * 255);
      g = 255;
      b = Math.floor((1 - intensity) * 2 * 255);
    }
    
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = Math.floor(predictionOpacity * 255);
  }
  
  offscreenCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(offscreen, 0, 0);
}

function drawPoint(x: number, y: number, color: string): void {
  if (!ctx) return;
  
  const radius = 5;
  
  // Draw outer circle (white border)
  ctx.beginPath();
  ctx.arc(x, y, radius + 2, 0, 2 * Math.PI);
  ctx.fillStyle = 'white';
  ctx.fill();
  
  // Draw inner circle (class color)
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

/**
 * Render the class list for selection
 */
export function renderSegmentationClasses(): void {
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
 * Handle class selection for segmentation
 */
export function handleSegmentationClassSelect(classname: string): void {
  const cls = classes.find(c => c.classname === classname);
  if (cls) {
    selectedClass = cls;
    renderSegmentationClasses();
  }
}

/**
 * Toggle point mode on/off
 */
export function togglePointMode(): void {
  isPointMode = !isPointMode;
  const btn = document.getElementById('point-mode-btn');
  if (btn) {
    if (isPointMode) {
      btn.classList.add('active');
      btn.textContent = '📍 Point Mode (Active)';
      if (canvas) {
        canvas.style.cursor = 'crosshair';
      }
    } else {
      btn.classList.remove('active');
      btn.textContent = '📍 Point Mode';
      if (canvas) {
        canvas.style.cursor = 'default';
      }
    }
  }
}

/**
 * Handle canvas click for adding/removing points
 */
export function handleCanvasClick(e: MouseEvent): void {
  if (!isPointMode || !canvas || !selectedClass) return;
  
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  
  if (e.button === 0) {
    // Left click - add point
    addPoint(x, y);
  } else if (e.button === 2) {
    // Right click - remove point
    removePoint(x, y);
  }
}

function addPoint(x: number, y: number): void {
  if (!selectedClass) return;
  
  points.push({
    type: 'point',
    classname: selectedClass.classname,
    color: selectedClass.color,
    x: x,
    y: y,
    origin: 'human',
  });
  
  redrawSegmentationCanvas();
}

function removePoint(x: number, y: number): void {
  const threshold = 10;
  
  let nearestIndex = -1;
  let nearestDist = threshold;
  
  points.forEach((point, index) => {
    if (point.x !== undefined && point.y !== undefined) {
      const dist = Math.sqrt(
        Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2)
      );
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestIndex = index;
      }
    }
  });
  
  if (nearestIndex >= 0) {
    points.splice(nearestIndex, 1);
    redrawSegmentationCanvas();
  }
}

function updatePointCount(): void {
  const countEl = document.getElementById('point-count');
  if (countEl) {
    countEl.textContent = `Points: ${points.length}`;
  }
}

/**
 * Set prediction opacity
 */
export function setOpacity(value: number): void {
  predictionOpacity = value / 100;
  redrawSegmentationCanvas();
}

/**
 * Toggle points visibility
 */
export function toggleShowPoints(): boolean {
  showPoints = !showPoints;
  redrawSegmentationCanvas();
  return showPoints;
}

/**
 * Toggle prediction visibility
 */
export function toggleShowPrediction(): boolean {
  showPrediction = !showPrediction;
  redrawSegmentationCanvas();
  return showPrediction;
}

/**
 * Load prediction mask from server
 */
export async function loadSegmentationPrediction(fileId: number): Promise<void> {
  if (!currentFile || fileId !== currentFile.id) {
    console.log('File changed, not loading prediction');
    return;
  }
  
  try {
    const response = await fetch(getPredictionUrl(fileId));
    if (!response.ok) {
      console.error('Failed to load prediction mask');
      return;
    }
    
    const blob = await response.blob();
    const maskImg = new Image();
    maskImg.src = URL.createObjectURL(blob);
    
    maskImg.onload = () => {
      if (!currentFile || fileId !== currentFile.id) {
        console.log('File changed during load, discarding prediction');
        URL.revokeObjectURL(maskImg.src);
        return;
      }
      
      predictionMask = maskImg;
      redrawSegmentationCanvas();
      console.log('Prediction mask loaded and displayed');
    };
    
    maskImg.onerror = () => {
      console.error('Error loading prediction mask image');
      URL.revokeObjectURL(maskImg.src);
    };
  } catch (error) {
    console.error('Error loading prediction mask:', error);
  }
}

/**
 * Handle receiving a new prediction from WebSocket
 */
export function onSegmentationPredictionReady(fileId: number): void {
  if (currentFile?.id === fileId) {
    loadSegmentationPrediction(fileId);
  }
}

/**
 * Handle extract points from prediction
 */
export async function handleExtractPoints(): Promise<boolean> {
  if (!currentFile) return false;
  
  const username = getUsername();
  if (!username) return false;
  
  try {
    const response = await fetch(`/api/files/${currentFile.id}/extract-points?created_by=${encodeURIComponent(username)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('Points extracted:', data);
      
      // Update points array with extracted points
      points = data.label_data.filter((item: any) => item.type === 'point');
      
      // Redraw canvas to show new points
      redrawSegmentationCanvas();
      
      return true;
    } else {
      const error = await response.json();
      throw new Error(error.detail || 'Extract points failed');
    }
  } catch (error) {
    console.error('Failed to extract points:', error);
    throw error;
  }
}

/**
 * Save segmentation labels
 */
export async function saveSegmentationLabels(): Promise<boolean> {
  if (!currentFile) return false;
  
  const username = getUsername();
  if (!username) return false;
  
  try {
    await saveLabels(currentFile.id, points, username);
    console.log('Segmentation labels saved');
    return true;
  } catch (error) {
    console.error('Failed to save segmentation labels:', error);
    return false;
  }
}

/**
 * Get current points for external access
 */
export function getCurrentPoints(): LabelDataItem[] {
  return points;
}

/**
 * Reset segmentation state
 */
export function resetSegmentationState(): void {
  currentFile = null;
  classes = [];
  selectedClass = null;
  points = [];
  isPointMode = false;
  canvas = null;
  ctx = null;
  img = null;
  predictionMask = null;
  predictionOpacity = 0.5;
  showPoints = true;
  showPrediction = true;
}

/**
 * Get current file ID
 */
export function getCurrentFileId(): number | null {
  return currentFile?.id || null;
}
