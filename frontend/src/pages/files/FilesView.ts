/**
 * Files page view - displays files list with statistics
 */

import { getUsername, clearUsername } from '../../shared/state';
import { getFiles, getFileStats, type FileItem, type FileStats } from '../../shared/api';

export async function renderFilesView(onLogout: () => void, onFileClick: (fileId: number) => void): Promise<void> {
  const mainContent = document.getElementById('main-content');
  if (!mainContent) return;

  // Show loading state
  mainContent.innerHTML = `
    <div class="files-container">
      <div class="files-header">
        <h2>Files</h2>
        <div class="user-info">
          <span class="username">${getUsername()}</span>
          <button id="logout-btn" class="btn-secondary">Logout</button>
        </div>
      </div>
      <div class="loading">Loading...</div>
    </div>
  `;

  try {
    // Fetch data from API
    const [files, stats] = await Promise.all([
      getFiles(),
      getFileStats(),
    ]);

    // Render the page with data
    mainContent.innerHTML = `
      <div class="files-container">
        <div class="files-header">
          <h2>Files</h2>
          <div class="user-info">
            <span class="username">${getUsername()}</span>
            <button id="logout-btn" class="btn-secondary">Logout</button>
          </div>
        </div>

        <div class="stats-section">
          <div class="stat-card">
            <div class="stat-label">Total Files</div>
            <div class="stat-value">${stats.total_files}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Manual Labeled</div>
            <div class="stat-value">${stats.manual_labeled}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Model Version</div>
            <div class="stat-value">${stats.model_version}</div>
          </div>
        </div>

        <div class="files-table-container">
          <table class="files-table">
            <thead>
              <tr>
                <th>Filename</th>
                <th>Filepath</th>
                <th>Labeled</th>
                <th>Labeler</th>
              </tr>
            </thead>
            <tbody>
              ${renderFileRows(files)}
            </tbody>
          </table>
        </div>
      </div>
    `;

    // Attach event listeners
    attachFilesListeners(onLogout, onFileClick);
  } catch (error) {
    console.error('Failed to load files:', error);
    mainContent.innerHTML = `
      <div class="files-container">
        <div class="error">Failed to load files. Please try again.</div>
      </div>
    `;
  }
}

function renderFileRows(files: FileItem[]): string {
  if (files.length === 0) {
    return `
      <tr>
        <td colspan="4" class="empty-message">No files found</td>
      </tr>
    `;
  }

  return files.map(file => `
    <tr class="file-row" data-file-id="${file.id}">
      <td class="filename-cell">${escapeHtml(file.filename)}</td>
      <td class="filepath">${escapeHtml(file.filepath)}</td>
      <td>
        <span class="label-badge label-${file.labeled}">${file.labeled}</span>
      </td>
      <td>${escapeHtml(file.labeler)}</td>
    </tr>
  `).join('');
}

function attachFilesListeners(onLogout: () => void, onFileClick: (fileId: number) => void): void {
  const logoutBtn = document.getElementById('logout-btn');
  
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      clearUsername();
      onLogout();
    });
  }
  
  // File row click handlers
  const fileRows = document.querySelectorAll('.file-row');
  fileRows.forEach(row => {
    row.addEventListener('click', () => {
      const fileId = parseInt((row as HTMLElement).dataset.fileId || '0');
      if (fileId > 0) {
        onFileClick(fileId);
      }
    });
  });
}

// Utility function to escape HTML
function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

