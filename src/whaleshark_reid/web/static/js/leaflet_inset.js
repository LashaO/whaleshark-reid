// Initialize a Leaflet map inset on the pair carousel.
// Reads lat/lon from data attrs on #map-inset element.
//
// HTMX replaces #pair-card.innerHTML, which destroys the #map-inset div
// without notifying Leaflet. The old map instance keeps DOM/window
// listeners, and panes (positioned absolute) can leak relative to the
// viewport. teardownMapInset() must run BEFORE the swap (htmx:beforeSwap)
// while the container is still attached to DOM — otherwise Leaflet's
// .remove() can't unbind cleanly.
let _currentMap = null;

function teardownMapInset() {
  if (_currentMap) {
    try { _currentMap.remove(); } catch (e) { /* container already gone */ }
    _currentMap = null;
  }
}

function initMapInset() {
  // Belt-and-suspenders: ensure no previous map is still bound (in case
  // teardown didn't fire — e.g., initial page load).
  teardownMapInset();

  const el = document.getElementById('map-inset');
  if (!el) return;
  const latA = parseFloat(el.dataset.latA);
  const lonA = parseFloat(el.dataset.lonA);
  const latB = parseFloat(el.dataset.latB);
  const lonB = parseFloat(el.dataset.lonB);

  // CRITICAL: force position:relative + overflow:hidden as INLINE styles
  // before L.map() runs. Without a positioned ancestor, Leaflet's
  // absolutely-positioned panes anchor to the viewport and tile images
  // render at (0,0) of the screen — covering the pair view.
  // Inline styles beat any cached/missing CSS rules.
  el.style.position = 'relative';
  el.style.overflow = 'hidden';

  // Clear placeholder content
  el.innerHTML = '';

  if (isNaN(latA) || isNaN(lonA) || isNaN(latB) || isNaN(lonB)) {
    el.innerHTML = '<span style="color:var(--text-muted);font-size:11px;">No GPS data</span>';
    return;
  }

  // No animations — render straight to the pair's bounds. Animations were
  // making the map appear to "fly in" during htmx swaps.
  const map = L.map(el, {
    zoomControl: false,
    attributionControl: false,
    fadeAnimation: false,
    zoomAnimation: false,
    markerZoomAnimation: false,
  });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);

  // DivIcon markers — pure CSS dots. Avoids vendoring the default leaflet
  // marker PNG assets (marker-icon*.png, marker-shadow.png) which weren't
  // included and 404 in this environment.
  const dotIcon = L.divIcon({
    className: 'map-dot',
    iconSize: [12, 12],
    iconAnchor: [6, 6],
  });
  L.marker([latA, lonA], { icon: dotIcon }).addTo(map);
  L.marker([latB, lonB], { icon: dotIcon }).addTo(map);
  L.polyline([[latA, lonA], [latB, lonB]], { color: '#00adb5', weight: 2 }).addTo(map);

  map.fitBounds([[latA, lonA], [latB, lonB]], { padding: [20, 20], animate: false });
  // Force a re-measure in case the container's layout settled after init.
  // Without this, panes can be positioned based on a stale 0×0 measurement.
  map.invalidateSize(false);
  _currentMap = map;
}

document.addEventListener('DOMContentLoaded', initMapInset);
