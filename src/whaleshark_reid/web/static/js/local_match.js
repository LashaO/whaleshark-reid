// Local feature match overlay for the pair card.
// Renders an SVG overlay spanning both chip images. On mount, tries GET
// /api/pairs/{id}/local-match; 200 → render, 404 → show "Run local match"
// button that POSTs. Controls for conf slider, line/keypoint toggles,
// click-to-hide, shift-click-to-hide-below.
(function () {
    const VIRIDIS = [
        [253, 231, 37], [180, 222, 44], [95, 201, 98],
        [33, 145, 140], [59, 82, 139], [68, 1, 84],
    ];
    function scoreColor(s) {
        const t = Math.max(0, Math.min(1, 1 - s)); // high score => warm end
        const i = t * (VIRIDIS.length - 1);
        const a = Math.floor(i), b = Math.min(a + 1, VIRIDIS.length - 1);
        const f = i - a;
        const c = VIRIDIS[a].map((v, k) => Math.round(v + (VIRIDIS[b][k] - v) * f));
        return `rgb(${c[0]},${c[1]},${c[2]})`;
    }

    function init(card) {
        const ctrl = card.querySelector(".local-match-controls");
        if (!ctrl || ctrl.dataset.initialized) return;
        ctrl.dataset.initialized = "1";
        const queueId = ctrl.dataset.queueId;
        const extractor = ctrl.dataset.extractor || "aliked";

        const overlay = card.querySelector(".local-match-overlay");
        const chipA = card.querySelector('.chip-wrap[data-chip="a"]');
        const chipB = card.querySelector('.chip-wrap[data-chip="b"]');
        const runBtn = ctrl.querySelector(".lm-run");
        const rerunBtn = ctrl.querySelector(".lm-rerun");
        const status = ctrl.querySelector(".lm-status");
        const stats = ctrl.querySelector(".lm-stats");
        const conf = ctrl.querySelector(".lm-conf");
        const confVal = ctrl.querySelector(".lm-conf-val");
        const linesCk = ctrl.querySelector(".lm-lines");
        const kptsCk = ctrl.querySelector(".lm-kpts");
        const toggleHidden = ctrl.querySelector(".lm-toggle-hidden");
        const hideBelow = ctrl.querySelector(".lm-hide-below");

        const state = { result: null, hidden: new Set() };

        function chipBox(wrap) {
            const img = wrap.querySelector("img");
            const wrapRect = wrap.getBoundingClientRect();
            const cardRect = card.getBoundingClientRect();
            return {
                x: wrapRect.left - cardRect.left,
                y: wrapRect.top - cardRect.top,
                w: img.clientWidth,
                h: img.clientHeight,
            };
        }

        function render() {
            if (!state.result) {
                overlay.innerHTML = "";
                return;
            }
            const r = state.result;
            const boxA = chipBox(chipA), boxB = chipBox(chipB);
            const sxA = boxA.w / r.img_a_size[0], syA = boxA.h / r.img_a_size[1];
            const sxB = boxB.w / r.img_b_size[0], syB = boxB.h / r.img_b_size[1];
            const cardRect = card.getBoundingClientRect();
            overlay.setAttribute("viewBox", `0 0 ${cardRect.width} ${cardRect.height}`);
            overlay.setAttribute("width", cardRect.width);
            overlay.setAttribute("height", cardRect.height);

            const thr = parseFloat(conf.value);
            const showLines = linesCk.checked;
            const showKpts = kptsCk.checked;

            let html = "";
            if (showLines) {
                r.matches.forEach((m, idx) => {
                    const [i, j, s] = m;
                    if (s < thr) return;
                    const pa = r.kpts_a[i], pb = r.kpts_b[j];
                    const x1 = boxA.x + pa[0] * sxA, y1 = boxA.y + pa[1] * syA;
                    const x2 = boxB.x + pb[0] * sxB, y2 = boxB.y + pb[1] * syB;
                    const cls = state.hidden.has(idx) ? "hidden" : "";
                    html += `<line class="${cls}" data-idx="${idx}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${scoreColor(s)}" stroke-width="1" stroke-opacity="0.9"/>`;
                });
            }
            if (showKpts) {
                const visible = new Set();
                r.matches.forEach((m, idx) => {
                    if (m[2] < thr || state.hidden.has(idx)) return;
                    visible.add(`a${m[0]}`); visible.add(`b${m[1]}`);
                });
                visible.forEach(k => {
                    const side = k[0], n = parseInt(k.slice(1), 10);
                    const p = side === "a" ? r.kpts_a[n] : r.kpts_b[n];
                    const box = side === "a" ? boxA : boxB;
                    const sx = side === "a" ? sxA : sxB, sy = side === "a" ? syA : syB;
                    html += `<circle cx="${box.x + p[0] * sx}" cy="${box.y + p[1] * sy}" r="2" fill="#fff" opacity="0.7"/>`;
                });
            }
            overlay.innerHTML = html;
            // Toggle button label reflects current state: empty → "Hide all",
            // something hidden → "Show all (N)".
            if (state.hidden.size) {
                toggleHidden.textContent = `Show all (${state.hidden.size})`;
            } else {
                toggleHidden.textContent = "Hide all";
            }
        }

        overlay.addEventListener("click", (e) => {
            const line = e.target.closest("line[data-idx]");
            if (!line || !state.result) return;
            const idx = parseInt(line.dataset.idx, 10);
            if (e.shiftKey) {
                const thisScore = state.result.matches[idx][2];
                state.result.matches.forEach((m, i) => {
                    if (m[2] <= thisScore) state.hidden.add(i);
                });
            } else {
                state.hidden.add(idx);
            }
            render();
        });

        toggleHidden.addEventListener("click", () => {
            if (!state.result) return;
            if (state.hidden.size) {
                state.hidden.clear();
            } else {
                state.result.matches.forEach((_m, i) => state.hidden.add(i));
            }
            render();
        });
        hideBelow.addEventListener("click", () => {
            if (!state.result) return;
            const thr = parseFloat(conf.value);
            state.result.matches.forEach((m, i) => {
                if (m[2] < thr) state.hidden.add(i);
            });
            render();
        });
        conf.addEventListener("input", () => { confVal.textContent = parseFloat(conf.value).toFixed(2); render(); });
        linesCk.addEventListener("change", render);
        kptsCk.addEventListener("change", render);

        function setStats(r) {
            stats.textContent = `${r.n_matches} matches @≥0.5 · mean ${(r.mean_score ?? 0).toFixed(2)}`;
        }

        function onResult(r) {
            state.result = r;
            state.hidden.clear();
            setStats(r);
            runBtn.style.display = "none";
            rerunBtn.style.display = "";
            toggleHidden.style.display = "";
            hideBelow.style.display = "";
            render();
        }

        async function runMatch(overwrite) {
            status.textContent = "…matching";
            runBtn.disabled = rerunBtn.disabled = true;
            try {
                const q = overwrite ? "?overwrite=1" : "";
                const r = await fetch(`/api/pairs/${queueId}/local-match${q}`, { method: "POST" });
                if (!r.ok) {
                    status.textContent = `error (${r.status})`;
                    return;
                }
                status.textContent = "";
                onResult(await r.json());
            } finally {
                runBtn.disabled = rerunBtn.disabled = false;
            }
        }

        runBtn.addEventListener("click", () => runMatch(false));
        rerunBtn.addEventListener("click", () => runMatch(true));

        const ro = new ResizeObserver(render);
        ro.observe(card);
        // HTMX replaces the #pair-card innerHTML on decide; the old .card node
        // becomes unreachable but the RO would keep it (and its render closure)
        // pinned. Disconnect before the swap completes.
        const onBeforeSwap = (e) => {
            if (!document.body.contains(card) || e.target.contains(card)) {
                ro.disconnect();
                document.body.removeEventListener("htmx:beforeSwap", onBeforeSwap);
            }
        };
        document.body.addEventListener("htmx:beforeSwap", onBeforeSwap);

        // Hybrid caching: fetch cached result on mount.
        fetch(`/api/pairs/${queueId}/local-match?extractor=${extractor}`).then(async (r) => {
            if (r.status === 200) {
                onResult(await r.json());
            } else {
                runBtn.style.display = "";
            }
        });
    }

    function initAll(root) {
        (root || document).querySelectorAll(".card[data-queue-id]").forEach(init);
    }

    document.addEventListener("DOMContentLoaded", () => initAll());
    document.body.addEventListener("htmx:afterSwap", (e) => initAll(e.target));
})();
