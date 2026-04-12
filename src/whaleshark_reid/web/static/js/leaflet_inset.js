// Initialize a Leaflet map inset on the pair carousel.
// Reads lat/lon from data attrs on #map-inset element.
function initMapInset() {
  const el = document.getElementById('map-inset');
  if (!el) return;
  const latA = parseFloat(el.dataset.latA);
  const lonA = parseFloat(el.dataset.lonA);
  const latB = parseFloat(el.dataset.latB);
  const lonB = parseFloat(el.dataset.lonB);

  // Clear previous content
  el.innerHTML = '';

  if (isNaN(latA) || isNaN(lonA) || isNaN(latB) || isNaN(lonB)) {
    el.innerHTML = '<span style="color:var(--text-muted);font-size:11px;">No GPS data</span>';
    return;
  }

  const map = L.map(el, { zoomControl: false, attributionControl: false });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);

  const mA = L.marker([latA, lonA]).addTo(map);
  const mB = L.marker([latB, lonB]).addTo(map);
  L.polyline([[latA, lonA], [latB, lonB]], { color: '#00adb5', weight: 2 }).addTo(map);

  map.fitBounds([[latA, lonA], [latB, lonB]], { padding: [20, 20] });
}

document.addEventListener('DOMContentLoaded', initMapInset);
