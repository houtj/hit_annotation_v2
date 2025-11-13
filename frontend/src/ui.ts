// UI rendering functions

export function renderUI() {
  const mainContent = document.getElementById('main-content');
  if (!mainContent) return;
  
  mainContent.innerHTML = `
    <div class="container">
      <p>Loading application...</p>
    </div>
  `;
}

