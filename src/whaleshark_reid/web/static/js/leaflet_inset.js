// Initialize a Leaflet map inset on the pair carousel.
// Reads lat/lon from data attrs on #map-inset element.
//
// HTMX swaps replace the #map-inset DOM node, but the old L.Map instance
// still holds DOM/window listeners. We track it in a module-level variable
// and call .remove() on each re-init to release the previous instance.
let _currentMap = null;

function initMapInset() {
  // Tear down any previous map instance from a prior swap.
  if (_currentMap) {
    try { _currentMap.remove(); } catch (e) { /* element may be detached */ }
    _currentMap = null;
  }

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

  // Disable all zoom/pan animations so the map renders instantly at the
  // pair's bounds — no pan from a default world view, no jump-out of
  // container during animation.
  const map = L.map(el, {
    zoomControl: false,
    attributionControl: false,
    fadeAnimation: false,
    zoomAnimation: false,
    markerZoomAnimation: false,
  });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);

  L.marker([latA, lonA]).addTo(map);
  L.marker([latB, lonB]).addTo(map);
  L.polyline([[latA, lonA], [latB, lonB]], { color: '#00adb5', weight: 2 }).addTo(map);

  map.fitBounds([[latA, lonA], [latB, lonB]], { padding: [20, 20], animate: false });
  _currentMap = map;
}

document.addEventListener('DOMContentLoaded', initMapInset);
