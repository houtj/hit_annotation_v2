/**
 * Labeling page view - orchestrates task-specific UI components
 * 
 * This view determines the task type (segmentation or classification) and
 * delegates to the appropriate UI component.
 */

import { getFile, getClasses, getCurrentVersion, getTrainingMetrics, type FileDetail, type Class } from '../../shared/api';
import { getTaskType } from '../../shared/state';

// Import task-specific UI components
import {
  renderSegmentationHTML,
  setupSegmentationCanvas,
  loadSegmentationImage,
  renderSegmentationClasses,
  handleSegmentationClassSelect,
  togglePointMode,
  handleCanvasClick,
  setOpacity,
  toggleShowPoints,
  toggleShowPrediction,
  loadSegmentationPrediction,
  onSegmentationPredictionReady,
  handleExtractPoints,
  saveSegmentationLabels,
  resetSegmentationState
} from './components/SegmentationUI';

import {
  renderClassificationHTML,
  setupClassificationCanvas,
  loadClassificationImage,
  renderClassificationClasses,
  handleClassificationClassSelect,
  loadClassificationPrediction,
  onClassificationPredictionReady,
  saveClassificationLabel,
  resetClassificationState
} from './components/ClassificationUI';

let currentFile: FileDetail | null = null;
let classes: Class[] = [];
let ws: WebSocket | null = null;
let wsReconnectTimer: number | null = null;
let wsActive: boolean = false;
let metricsChart: any = null;
let metricsData: Array<{epoch: number, train_loss: number, test_loss: number}> = [];
let currentMajorVersion: number = 0;

export async function renderLabelingView(fileId: number, onBack: () => void): Promise<void> {
  const mainContent = document.getElementById('main-content');
  if (!mainContent) return;

  // Show loading
  mainContent.innerHTML = '<div class="loading">Loading...</div>';

  try {
    // Fetch data
    [currentFile, classes] = await Promise.all([
      getFile(fileId),
      getClasses(),
    ]);

    // Get task type from app state
    const taskType = getTaskType();
    
    // Render task-specific HTML
    if (taskType === 'classification') {
      mainContent.innerHTML = renderClassificationHTML(currentFile, classes);
    } else {
      mainContent.innerHTML = renderSegmentationHTML(currentFile, classes);
    }

    // Setup task-specific canvas and listeners
    if (taskType === 'classification') {
      setupClassificationCanvas();
      renderClassificationClasses();
      attachClassificationListeners(onBack);
      await loadClassificationImage();
      // Try to load existing prediction
      await loadClassificationPrediction(fileId);
    } else {
      setupSegmentationCanvas();
      renderSegmentationClasses();
      attachSegmentationListeners(onBack);
      await loadSegmentationImage();
      // Try to load existing prediction
      tryLoadExistingPrediction(fileId);
    }
    
    // Connect WebSocket for real-time updates (shared between tasks)
    wsActive = true;
    connectWebSocket(taskType);
    
    // Initialize metrics chart (shared between tasks)
    await initializeMetricsChart();

  } catch (error) {
    console.error('Failed to load labeling page:', error);
    mainContent.innerHTML = '<div class="error">Failed to load file</div>';
  }
}

// ============================================================================
// Segmentation Event Listeners
// ============================================================================

function attachSegmentationListeners(onBack: () => void): void {
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
  
  // Extract points button
  const extractPointsBtn = document.getElementById('extract-points-btn');
  if (extractPointsBtn) {
    extractPointsBtn.addEventListener('click', async () => {
      const btn = extractPointsBtn as HTMLButtonElement;
      btn.disabled = true;
      btn.textContent = '🎯 Extracting...';
      
      try {
        await handleExtractPoints();
        btn.textContent = '✓ Extracted';
        setTimeout(() => {
          btn.textContent = '🎯 Extract Points';
          btn.disabled = false;
        }, 2000);
      } catch (error: any) {
        alert(`Failed to extract points: ${error.message || error}`);
        btn.textContent = '🎯 Extract Points';
        btn.disabled = false;
      }
    });
  }
  
  // Save button
  const saveBtn = document.getElementById('save-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      await handleSegmentationSave(saveBtn as HTMLButtonElement);
    });
  }
  
  // Stop button
  const stopBtn = document.getElementById('stop-btn');
  if (stopBtn) {
    stopBtn.addEventListener('click', handleStop);
  }
  
  // Canvas interactions
  const canvas = document.getElementById('annotation-canvas') as HTMLCanvasElement;
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
        if (classname) {
          handleSegmentationClassSelect(classname);
        }
      }
    });
  }
  
  // Opacity slider
  const opacitySlider = document.getElementById('opacity-slider') as HTMLInputElement;
  const opacityValue = document.getElementById('opacity-value');
  if (opacitySlider && opacityValue) {
    opacitySlider.addEventListener('input', (e) => {
      const value = parseInt((e.target as HTMLInputElement).value);
      opacityValue.textContent = `${value}%`;
      setOpacity(value);
    });
  }
  
  // Toggle points visibility
  const togglePointsBtn = document.getElementById('toggle-points-btn');
  if (togglePointsBtn) {
    togglePointsBtn.addEventListener('click', () => {
      const showPoints = toggleShowPoints();
      togglePointsBtn.classList.toggle('active', showPoints);
      togglePointsBtn.innerHTML = showPoints 
        ? '<span class="toggle-icon">👁️</span> Show Points'
        : '<span class="toggle-icon">🚫</span> Hide Points';
    });
  }
  
  // Toggle prediction visibility
  const togglePredictionBtn = document.getElementById('toggle-prediction-btn');
  if (togglePredictionBtn) {
    togglePredictionBtn.addEventListener('click', () => {
      const showPrediction = toggleShowPrediction();
      togglePredictionBtn.classList.toggle('active', showPrediction);
      togglePredictionBtn.innerHTML = showPrediction 
        ? '<span class="toggle-icon">👁️</span> Show Prediction'
        : '<span class="toggle-icon">🚫</span> Hide Prediction';
    });
  }
}

async function handleSegmentationSave(saveBtn: HTMLButtonElement): Promise<void> {
  saveBtn.disabled = true;
  saveBtn.textContent = '💾 Saving...';
  
  try {
    await saveSegmentationLabels();
    
    // Trigger training
    try {
      const response = await fetch(`/api/training/start?file_id=${currentFile?.id}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Training started:', data.version);
        saveBtn.textContent = '✓ Saved & Training';
      } else {
        saveBtn.textContent = '✓ Saved (Training failed)';
      }
    } catch (trainingError) {
      console.error('Failed to start training:', trainingError);
      saveBtn.textContent = '✓ Saved (Training failed)';
    }
    
    setTimeout(() => {
      saveBtn.textContent = '💾 Save';
      saveBtn.disabled = false;
    }, 2000);
  } catch (error) {
    console.error('Failed to save labels:', error);
    alert('Failed to save labels');
    saveBtn.textContent = '💾 Save';
    saveBtn.disabled = false;
  }
}

// ============================================================================
// Classification Event Listeners
// ============================================================================

function attachClassificationListeners(onBack: () => void): void {
  // Back button
  const backBtn = document.getElementById('back-btn');
  if (backBtn) {
    backBtn.addEventListener('click', onBack);
  }
  
  // Save button
  const saveBtn = document.getElementById('save-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      await handleClassificationSave(saveBtn as HTMLButtonElement);
    });
  }
  
  // Stop button
  const stopBtn = document.getElementById('stop-btn');
  if (stopBtn) {
    stopBtn.addEventListener('click', handleStop);
  }
  
  // Class selection
  const classList = document.getElementById('classes-list');
  if (classList) {
    classList.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const classItem = target.closest('.class-item') as HTMLElement;
      if (classItem) {
        const classname = classItem.dataset.classname;
        if (classname) {
          handleClassificationClassSelect(classname);
        }
      }
    });
  }
}

async function handleClassificationSave(saveBtn: HTMLButtonElement): Promise<void> {
  saveBtn.disabled = true;
  saveBtn.textContent = '💾 Saving...';
  
  try {
    const saved = await saveClassificationLabel();
    
    if (saved) {
      // Trigger training
      try {
        const response = await fetch(`/api/training/start?file_id=${currentFile?.id}`, {
          method: 'POST'
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Training started:', data.version);
          saveBtn.textContent = '✓ Saved & Training';
        } else {
          saveBtn.textContent = '✓ Saved (Training failed)';
        }
      } catch (trainingError) {
        console.error('Failed to start training:', trainingError);
        saveBtn.textContent = '✓ Saved (Training failed)';
      }
    } else {
      saveBtn.textContent = 'No label to save';
    }
    
    setTimeout(() => {
      saveBtn.textContent = '💾 Save';
      saveBtn.disabled = false;
    }, 2000);
  } catch (error) {
    console.error('Failed to save label:', error);
    alert('Failed to save label');
    saveBtn.textContent = '💾 Save';
    saveBtn.disabled = false;
  }
}

// ============================================================================
// Shared Functions
// ============================================================================

async function handleStop(): Promise<void> {
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

function connectWebSocket(taskType: string): void {
  if (!wsActive) {
    console.log('WebSocket not active, skipping connection');
    return;
  }
  
  if (ws) {
    ws.close();
  }
  
  if (wsReconnectTimer !== null) {
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = null;
  }
  
  ws = new WebSocket('ws://localhost:8000/ws');
  
  ws.onopen = () => {
    console.log('WebSocket connected');
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === 'prediction_ready') {
        if (currentFile && data.file_id === currentFile.id) {
          console.log('Prediction ready for current file, version:', data.version);
          
          // Dispatch to appropriate handler based on task type
          if (taskType === 'classification') {
            onClassificationPredictionReady(data.file_id);
          } else {
            onSegmentationPredictionReady(data.file_id);
          }
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
    if (wsActive) {
      console.log('Reconnecting in 5 seconds...');
      wsReconnectTimer = window.setTimeout(() => connectWebSocket(taskType), 5000);
    } else {
      console.log('WebSocket deactivated, not reconnecting');
    }
  };
}

export function cleanupLabelingView(): void {
  wsActive = false;
  
  if (wsReconnectTimer !== null) {
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = null;
  }
  
  if (ws) {
    ws.close();
    ws = null;
  }
  
  // Reset task-specific state
  resetSegmentationState();
  resetClassificationState();
  
  // Reset shared state
  currentFile = null;
  classes = [];
  metricsData = [];
  currentMajorVersion = 0;
  
  if (metricsChart) {
    metricsChart.destroy();
    metricsChart = null;
  }
  
  console.log('Labeling view cleaned up');
}

async function tryLoadExistingPrediction(fileId: number): Promise<void> {
  if (!currentFile || fileId !== currentFile.id) {
    return;
  }
  
  try {
    await loadSegmentationPrediction(fileId);
  } catch (error) {
    console.log('No existing prediction available');
  }
}

function updateTrainingProgress(data: any): void {
  const incomingMajorVersion = parseInt(data.version.split('.')[0]);
  
  if (incomingMajorVersion !== currentMajorVersion) {
    console.log(`New major version detected: ${incomingMajorVersion} (was ${currentMajorVersion})`);
    currentMajorVersion = incomingMajorVersion;
    metricsData = [];
    
    if (metricsChart) {
      metricsChart.data.labels = [];
      metricsChart.data.datasets[0].data = [];
      metricsChart.data.datasets[1].data = [];
    }
  }
  
  metricsData.push({
    epoch: data.epoch,
    train_loss: data.train_loss,
    test_loss: data.test_loss
  });
  
  if (metricsChart) {
    metricsChart.data.labels.push(data.epoch);
    metricsChart.data.datasets[0].data.push(data.train_loss);
    metricsChart.data.datasets[1].data.push(data.test_loss);
    metricsChart.update('none');
  }
  
  const trainEl = document.getElementById('latest-train');
  const testEl = document.getElementById('latest-test');
  if (trainEl) trainEl.textContent = data.train_loss.toFixed(4);
  if (testEl) testEl.textContent = data.test_loss.toFixed(4);
}

async function initializeMetricsChart(): Promise<void> {
  try {
    const data = await getCurrentVersion();
    const versionStr = data.version || '0.0';
    const majorStr = versionStr.split('.')[0] || '0';
    currentMajorVersion = parseInt(majorStr);
  } catch (error) {
    console.log('Could not fetch current version, defaulting to 0');
    currentMajorVersion = 0;
  }
  
  if (currentMajorVersion > 0) {
    try {
      metricsData = await getTrainingMetrics(currentMajorVersion);
    } catch (error) {
      console.error('Error loading metrics:', error);
      metricsData = [];
    }
  }
  
  const chartCanvas = document.getElementById('metrics-chart') as HTMLCanvasElement;
  if (!chartCanvas) return;
  
  if (metricsChart) {
    metricsChart.destroy();
  }
  
  // @ts-ignore - Chart is loaded from CDN
  metricsChart = new Chart(chartCanvas, {
    type: 'line',
    data: {
      labels: metricsData.map(m => m.epoch),
      datasets: [
        {
          label: 'Train Loss',
          data: metricsData.map(m => m.train_loss),
          borderColor: '#e74c3c',
          backgroundColor: 'rgba(231, 76, 60, 0.1)',
          tension: 0.1,
        },
        {
          label: 'Test Loss',
          data: metricsData.map(m => m.test_loss),
          borderColor: '#3498db',
          backgroundColor: 'rgba(52, 152, 219, 0.1)',
          tension: 0.1,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2,
      plugins: {
        legend: { 
          position: 'top',
          labels: { font: { size: 11 } }
        },
        title: { 
          display: false
        }
      },
      scales: {
        x: { 
          title: { display: true, text: 'Epoch', font: { size: 11 } },
          ticks: { font: { size: 10 } }
        },
        y: { 
          title: { display: true, text: 'Loss', font: { size: 11 } },
          beginAtZero: false,
          ticks: { font: { size: 10 } }
        }
      }
    }
  });
  
  if (metricsData.length > 0) {
    const latest = metricsData[metricsData.length - 1];
    if (latest) {
      const trainEl = document.getElementById('latest-train');
      const testEl = document.getElementById('latest-test');
      if (trainEl) trainEl.textContent = latest.train_loss.toFixed(4);
      if (testEl) testEl.textContent = latest.test_loss.toFixed(4);
    }
  }
}
