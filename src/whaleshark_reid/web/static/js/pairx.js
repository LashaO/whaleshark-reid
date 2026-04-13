// PairX A/B viz — server renders a PNG, we just <img> it.
// On mount: HEAD the cached PNG; if present render it, else show button.
(function () {
    function init(card) {
        const sec = card.querySelector(".pairx-section");
        if (!sec || sec.dataset.initialized) return;
        sec.dataset.initialized = "1";

        const queueId = sec.dataset.queueId;
        const layer = sec.dataset.layer || "backbone.blocks.3";
        const status = sec.querySelector(".px-status");
        const runBtn = sec.querySelector(".px-run");
        const rerunBtn = sec.querySelector(".px-rerun");
        const wrap = sec.querySelector(".px-img-wrap");

        const url = `/api/pairs/${queueId}/pairx.png?layer=${encodeURIComponent(layer)}`;

        function showImg(srcWithBust) {
            wrap.innerHTML = `<img src="${srcWithBust}" alt="PairX visualization" style="max-width:100%; display:block; border:1px solid var(--border); border-radius:4px;">`;
            runBtn.style.display = "none";
            rerunBtn.style.display = "";
            status.textContent = "";
        }

        async function run(force) {
            status.textContent = "…computing (slow)";
            runBtn.disabled = rerunBtn.disabled = true;
            try {
                const r = await fetch(url, { method: "POST" });
                if (!r.ok) {
                    status.textContent = `error (${r.status})`;
                    return;
                }
                showImg(`${url}&t=${Date.now()}`);
            } finally {
                runBtn.disabled = rerunBtn.disabled = false;
            }
        }

        runBtn.addEventListener("click", () => run(false));
        rerunBtn.addEventListener("click", () => run(true));

        // Hybrid caching: HEAD the PNG; if it exists, render. If not, show the button.
        fetch(url, { method: "HEAD" }).then((r) => {
            if (r.status === 200) showImg(url);
            else runBtn.style.display = "";
        });
    }

    function initAll(root) {
        (root || document).querySelectorAll(".card[data-queue-id]").forEach(init);
    }

    document.addEventListener("DOMContentLoaded", () => initAll());
    document.body.addEventListener("htmx:afterSwap", (e) => initAll(e.target));
})();
