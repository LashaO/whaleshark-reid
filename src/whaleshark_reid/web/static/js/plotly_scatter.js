// Fetch projection data and render a Plotly scatter colored by cluster label.
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('scatter');
  if (!container) return;
  const url = container.dataset.projectionUrl;
  fetch(url)
    .then(r => r.json())
    .then(data => {
      if (!data.points || data.points.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">No projection data. Run <code>catalog project</code>.</p>';
        return;
      }
      const x = data.points.map(p => p.x);
      const y = data.points.map(p => p.y);
      const colors = data.points.map(p => p.cluster_label);
      const text = data.points.map(p => `${p.annotation_uuid.slice(0,12)}… (cluster ${p.cluster_label})`);
      Plotly.newPlot(container, [{
        x, y, mode: 'markers', type: 'scatter',
        marker: { color: colors, colorscale: 'Portland', size: 8 },
        text, hoverinfo: 'text',
      }], {
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: { color: '#e0e0e0', family: 'monospace' },
        margin: { t: 20, b: 40, l: 40, r: 20 },
        xaxis: { gridcolor: '#333' }, yaxis: { gridcolor: '#333' },
      }, { responsive: true });
    });
});
