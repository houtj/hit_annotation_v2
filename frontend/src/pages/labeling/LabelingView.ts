/**
 * Labeling page view - annotate images with points
 */

import { getFile, getClasses, getImageUrl, saveLabels, type FileDetail, type Class, type LabelDataItem } from '../../shared/api';
import { getUsername } from '../../shared/state';

let currentFile: FileDetail | null = null;
let classes: Class[] = [];
let selectedClass: Class | null = null;
let points: LabelDataItem[] = [];
let isPointMode = false;
let canvas: HTMLCanvasElement | null = null;
let ctx: CanvasRenderingContext2D | null = null;
let img: HTMLImageElement | null = null;
let ws: WebSocket | null = null;
let predictionMask: HTMLImageElement | null = null;
let predictionOpacity: number = 0.5;

export async function renderLabelingView(fileId: number, onBack: () => void): Promise<void> {
  const mainContent = document.getElementById('main-content');
  if (!mainContent) return;

  // Reset prediction mask when changing files
  predictionMask = null;

  // Show loading
  mainContent.innerHTML = '<div class="loading">Loading...</div>';

  try {
    // Fetch data
    [currentFile, classes] = await Promise.all([
      getFile(fileId),
      getClasses(),
    ]);

    // Set first class as default
    if (classes.length > 0) {
      selectedClass = classes[0];
    }

    // Load existing labels
    if (currentFile.label) {
      points = currentFile.label.label_data.filter(item => item.type === 'point');
    } else {
      points = [];
    }

    // Render page
    mainContent.innerHTML = `
      <div class="labeling-container">
        <div class="labeling-header">
          <button id="back-btn" class="btn-back">← Back</button>
          <h2>${escapeHtml(currentFile.filename)}</h2>
          <div class="header-spacer"></div>
        </div>
        
        <div class="labeling-body">
          <div class="left-panel">
            <div class="toolbar">
              <button id="point-mode-btn" class="tool-btn">📍 Point Mode</button>
              <button id="save-btn" class="btn-primary">💾 Save</button>
              <button id="stop-btn" class="btn-secondary">⏹ Stop Training</button>
              <span id="point-count" class="point-count">Points: 0</span>
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
            
            <div class="prediction-section">
              <h3>Prediction Overlay</h3>
              <div class="opacity-control">
                <label for="opacity-slider">Opacity: <span id="opacity-value">50%</span></label>
                <input type="range" id="opacity-slider" min="0" max="100" value="50" />
              </div>
            </div>
            
            <div class="progress-section">
              <h3>Training Progress</h3>
              <div class="progress-placeholder">
                <p>Training not started</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    // Setup canvas
    setupCanvas();
    
    // Render classes
    renderClasses();
    
    // Attach event listeners
    attachLabelingListeners(onBack);
    
    // Connect WebSocket
    connectWebSocket();
    
    // Load image
    await loadImage();
    
    // Try to load existing prediction if available
    tryLoadExistingPrediction(fileId);

  } catch (error) {
    console.error('Failed to load labeling page:', error);
    mainContent.innerHTML = '<div class="error">Failed to load file</div>';
  }
}

function setupCanvas() {
  canvas = document.getElementById('annotation-canvas') as HTMLCanvasElement;
  if (!canvas || !currentFile) return;
  
  ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // Set canvas size to image size
  canvas.width = currentFile.width;
  canvas.height = currentFile.height;
}

async function loadImage() {
  if (!canvas || !ctx || !currentFile) return;
  
  img = new Image();
  img.crossOrigin = 'anonymous';
  
  img.onload = () => {
    redrawCanvas();
  };
  
  img.src = getImageUrl(currentFile.id);
}

function redrawCanvas() {
  if (!canvas || !ctx || !img) return;
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw image (layer 1: bottom)
  ctx.drawImage(img, 0, 0);
  
  // Draw prediction mask if available (layer 2: above image, below points)
  if (predictionMask) {
    ctx.globalAlpha = predictionOpacity;
    ctx.drawImage(predictionMask, 0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 1.0;
  }
  
  // Draw points (layer 3: on top)
  points.forEach(point => {
    if (point.x !== undefined && point.y !== undefined) {
      drawPoint(point.x, point.y, point.color);
    }
  });
  
  // Update point count
  updatePointCount();
}

function drawPoint(x: number, y: number, color: string) {
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

function renderClasses() {
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

function attachLabelingListeners(onBack: () => void) {
  // Back button
  const backBtn = document.getElementById('back-btn');
  if (backBtn) {
    backBtn.addEventListener('click', onBack);
  }
  
  // Point mode button
  const pointModeBtn = document.getElementById('point-mode-btn');
  if (pointModeBtn) {
    pointModeBtn.addEventListener('click', togglePointMode);
  }
  
  // Save button
  const saveBtn = document.getElementById('save-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', handleSave);
  }
  
  // Stop button
  const stopBtn = document.getElementById('stop-btn');
  if (stopBtn) {
    stopBtn.addEventListener('click', handleStop);
  }
  
  // Canvas interactions
  if (canvas) {
    canvas.addEventListener('mousedown', handleCanvasClick);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }
  
  // Class selection
  const classList = document.getElementById('classes-list');
  if (classList) {
    classList.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const classItem = target.closest('.class-item') as HTMLElement;
      if (classItem) {
        const classname = classItem.dataset.classname;
        selectedClass = classes.find(c => c.classname === classname) || null;
        renderClasses();
      }
    });
  }
  
  // Opacity slider
  const opacitySlider = document.getElementById('opacity-slider') as HTMLInputElement;
  const opacityValue = document.getElementById('opacity-value');
  if (opacitySlider && opacityValue) {
    opacitySlider.addEventListener('input', (e) => {
      const value = parseInt((e.target as HTMLInputElement).value);
      predictionOpacity = value / 100;
      opacityValue.textContent = `${value}%`;
      redrawCanvas();
    });
  }
}

function togglePointMode() {
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

function handleCanvasClick(e: MouseEvent) {
  if (!isPointMode || !canvas || !selectedClass) return;
  
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  if (e.button === 0) {
    // Left click - add point
    addPoint(x, y);
  } else if (e.button === 2) {
    // Right click - remove point
    removePoint(x, y);
  }
}

function addPoint(x: number, y: number) {
  if (!selectedClass) return;
  
  points.push({
    type: 'point',
    classname: selectedClass.classname,
    color: selectedClass.color,
    x: x,
    y: y,
  });
  
  redrawCanvas();
}

function removePoint(x: number, y: number) {
  const threshold = 10;
  
  // Find nearest point within threshold
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
    redrawCanvas();
  }
}

function updatePointCount() {
  const countEl = document.getElementById('point-count');
  if (countEl) {
    countEl.textContent = `Points: ${points.length}`;
  }
}

async function handleSave() {
  if (!currentFile) return;
  
  const username = getUsername();
  if (!username) return;
  
  try {
    const saveBtn = document.getElementById('save-btn') as HTMLButtonElement;
    if (saveBtn) {
      saveBtn.disabled = true;
      saveBtn.textContent = '💾 Saving...';
    }
    
    // Save labels
    await saveLabels(currentFile.id, points, username);
    
    // Trigger training
    try {
      const response = await fetch(`/api/training/start?file_id=${currentFile.id}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Training started:', data.version);
        if (saveBtn) {
          saveBtn.textContent = '✓ Saved & Training';
        }
      } else {
        if (saveBtn) {
          saveBtn.textContent = '✓ Saved (Training failed)';
        }
      }
    } catch (trainingError) {
      console.error('Failed to start training:', trainingError);
      if (saveBtn) {
        saveBtn.textContent = '✓ Saved (Training failed)';
      }
    }
    
    if (saveBtn) {
      setTimeout(() => {
        saveBtn.textContent = '💾 Save';
        saveBtn.disabled = false;
      }, 2000);
    }
  } catch (error) {
    console.error('Failed to save labels:', error);
    alert('Failed to save labels');
  }
}

async function handleStop() {
  try {
    const response = await fetch('/api/training/stop', {
      method: 'POST'
    });
    
    if (response.ok) {
      console.log('Training stop signal sent');
      alert('Training stop signal sent');
    } else {
      alert('Failed to send stop signal');
    }
  } catch (error) {
    console.error('Failed to stop training:', error);
    alert('Failed to stop training');
  }
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function connectWebSocket() {
  // Close existing connection if any
  if (ws) {
    ws.close();
  }
  
  // Connect to WebSocket
  ws = new WebSocket('ws://localhost:8000/ws');
  
  ws.onopen = () => {
    console.log('WebSocket connected');
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === 'prediction_ready') {
        // Only load if we're still viewing the same file
        if (currentFile && data.file_id === currentFile.id) {
          console.log('Prediction ready for current file, version:', data.version);
          loadPredictionMask(data.file_id);
        } else {
          console.log('Prediction ready for different file, ignoring');
        }
      } else if (data.type === 'training_progress') {
        console.log('Training progress:', data);
        updateTrainingProgress(data);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket closed');
    // Attempt to reconnect after 5 seconds
    setTimeout(connectWebSocket, 5000);
  };
}

async function loadPredictionMask(fileId: number) {
  // Double-check we're still on the same file
  if (!currentFile || fileId !== currentFile.id) {
    console.log('File changed, not loading prediction');
    return;
  }
  
  try {
    const response = await fetch(`/api/files/${fileId}/prediction`);
    if (!response.ok) {
      console.error('Failed to load prediction mask');
      return;
    }
    
    const blob = await response.blob();
    const maskImg = new Image();
    maskImg.src = URL.createObjectURL(blob);
    
    maskImg.onload = () => {
      // Final check: ensure we're still on the same file
      if (!currentFile || fileId !== currentFile.id) {
        console.log('File changed during load, discarding prediction');
        URL.revokeObjectURL(maskImg.src);
        return;
      }
      
      // Store the prediction mask
      predictionMask = maskImg;
      
      // Redraw canvas with the new prediction
      redrawCanvas();
      
      console.log('Prediction mask loaded and displayed');
      
      // Note: Don't revoke the URL yet, we need it for redrawing
      // It will be cleaned up when a new prediction is loaded
    };
    
    maskImg.onerror = () => {
      console.error('Error loading prediction mask image');
      URL.revokeObjectURL(maskImg.src);
    };
  } catch (error) {
    console.error('Error loading prediction mask:', error);
  }
}

function updateTrainingProgress(data: any) {
  const progressSection = document.querySelector('.progress-section .progress-placeholder');
  if (progressSection) {
    progressSection.innerHTML = `
      <p><strong>Version:</strong> ${data.version}</p>
      <p><strong>Epoch:</strong> ${data.epoch}</p>
      <p><strong>Loss:</strong> ${data.loss.toFixed(4)}</p>
      ${data.accuracy ? `<p><strong>Accuracy:</strong> ${data.accuracy.toFixed(4)}</p>` : ''}
    `;
  }
}

async function tryLoadExistingPrediction(fileId: number) {
  // Try to load existing prediction without throwing errors
  if (!currentFile || fileId !== currentFile.id) {
    return;
  }
  
  try {
    await loadPredictionMask(fileId);
  } catch (error) {
    // Silently ignore if no prediction exists yet
    console.log('No existing prediction available');
  }
}

