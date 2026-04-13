// Keyboard shortcut wiring for the pair review carousel.
// Reads data-shortcut attrs from buttons and clicks them on keypress.
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const key = e.key.toLowerCase();

  // J/K for prev/next navigation
  if (key === 'j') {
    const prev = document.querySelector('a[href*="position="]');
    if (prev && prev.textContent.includes('Prev')) { prev.click(); e.preventDefault(); }
    return;
  }
  if (key === 'k') {
    const links = document.querySelectorAll('a[href*="position="]');
    const next = Array.from(links).find(a => a.textContent.includes('Next'));
    if (next) { next.click(); e.preventDefault(); }
    return;
  }

  // Decision shortcuts: Y, N, U, Space
  const btn = document.querySelector(`[data-shortcut="${key}"]`);
  if (btn) { btn.click(); e.preventDefault(); }
});

// Tear down the previous Leaflet map BEFORE htmx replaces the DOM, so we can
// .remove() it while its container is still attached. After the swap, init a
// new map on the freshly-mounted #map-inset element. We defer the init one
// frame so the browser flushes layout first — otherwise Leaflet measures a
// 0×0 container and the tile pane positions itself relative to the viewport
// (top-left), spilling out and blocking the pair view.
document.body.addEventListener('htmx:beforeSwap', (e) => {
  // Only teardown if the swap will replace the container holding the map.
  const target = e.detail.target;
  if (target && target.contains(document.getElementById('map-inset'))) {
    if (typeof teardownMapInset === 'function') teardownMapInset();
  }
});
document.body.addEventListener('htmx:afterSwap', (e) => {
  const target = e.detail.target;
  if (target && target.querySelector('#map-inset')) {
    if (typeof initMapInset === 'function') {
      requestAnimationFrame(() => initMapInset());
    }
  }
});
