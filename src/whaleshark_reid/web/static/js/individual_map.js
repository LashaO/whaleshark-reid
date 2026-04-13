// Multi-point Leaflet map for the individual detail page.
// Reads GPS points from the data-points JSON attr on #individual-map.
function initIndividualMap() {
  const el = document.getElementById('individual-map');
  if (!el) return;

  let points;
  try { points = JSON.parse(el.dataset.points || '[]'); }
  catch (e) { console.error('individual-map: invalid JSON in data-points', e); return; }

  if (!points.length) return;

  // Force containing-block behavior — same defensive inline styles as the
  // pair carousel map (see leaflet_inset.js for the rationale).
  el.style.position = 'relative';
  el.style.overflow = 'hidden';
  el.innerHTML = '';

  const map = L.map(el, {
    fadeAnimation: false,
    zoomAnimation: false,
    markerZoomAnimation: false,
  });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);

  const dotIcon = L.divIcon({
    className: 'map-dot',
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });

  const latlngs = [];
  points.forEach(p => {
    if (p.lat == null || p.lon == null) return;
    const marker = L.marker([p.lat, p.lon], { icon: dotIcon }).addTo(map);
    const label = `${p.uuid.slice(0, 8)}…${p.date ? ` · ${p.date}` : ''}`;
    marker.bindTooltip(label, { direction: 'top', offset: [0, -6] });
    latlngs.push([p.lat, p.lon]);
  });

  if (latlngs.length >= 2) {
    // Polyline connecting sightings in capture-date order
    L.polyline(latlngs, { color: '#00adb5', weight: 2, opacity: 0.6 }).addTo(map);
    map.fitBounds(latlngs, { padding: [30, 30], animate: false });
  } else {
    map.setView(latlngs[0], 10);
  }
  map.invalidateSize(false);
}

document.addEventListener('DOMContentLoaded', initIndividualMap);
