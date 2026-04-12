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

// After every HTMX swap, re-initialize the Leaflet map inset on the new pair card.
// Without this, the map shows "map loading…" after any decision click.
document.body.addEventListener('htmx:afterSwap', () => {
  if (typeof initMapInset === 'function') {
    initMapInset();
  }
});
