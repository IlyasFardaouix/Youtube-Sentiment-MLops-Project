let predictions = [];
let lastStats = null;
const $ = (id) => document.getElementById(id);

const SENTIMENTS = {
  "1": { label: "Positifs", color: "#3f8f64" },
  "0": { label: "Neutres", color: "#b08a43" },
  "-1": { label: "Négatifs", color: "#c64f45" },
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function render(filter = "all") {
  $("results").innerHTML = predictions
    .filter((item) => filter === "all" || String(item.label) === filter)
    .map((item) => {
      const confidence = item.confidence == null ? "" : `${(item.confidence * 100).toFixed(1)}%`;
      return `<li class="item ${item.sentiment}"><strong>${item.sentiment}</strong> ${confidence}<br>${escapeHtml(item.text)}</li>`;
    })
    .join("");
}

function renderChart(stats) {
  const total = stats.total || 1;
  $("chart").innerHTML = ["1", "0", "-1"]
    .map((label) => {
      const meta = SENTIMENTS[label];
      const count = stats.counts[label] || 0;
      const pct = ((count / total) * 100).toFixed(1);
      return `<div class="bar-row"><span class="bar-label">${meta.label}</span><span class="bar-track"><span class="bar-fill" style="width:${pct}%;background:${meta.color}"></span></span><span class="bar-value">${pct}% (${count})</span></div>`;
    })
    .join("");
}

function setLoading(active) {
  $("analyze").disabled = active;
  document.body.classList.toggle("loading", active);
}

$("theme").onclick = () => document.body.classList.toggle("dark");
document.querySelectorAll("[data-filter]").forEach((button) => {
  button.onclick = () => render(button.dataset.filter);
});

$("analyze").onclick = async () => {
  setLoading(true);
  $("status").textContent = "Analyse en cours...";
  try {
    const tab = (await chrome.tabs.query({ active: true, currentWindow: true }))[0];
    const extracted = await chrome.tabs.sendMessage(tab.id, { type: "extractComments" });
    if (!extracted || !extracted.comments.length) throw new Error("Aucun commentaire visible");
    const base = $("api").value.trim().replace(/\/+$/, "");
    const response = await fetch(`${base}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: extracted.comments }),
    });
    if (!response.ok) throw new Error(`API ${response.status}`);
    const data = await response.json();
    predictions = data.predictions;
    lastStats = data.statistics;
    renderChart(lastStats);
    const total = lastStats.total;
    const percent = (label) => ((lastStats.counts[label] / total) * 100).toFixed(1);
    $("stats").textContent = `${total} commentaires | Positifs ${percent("1")}% | Neutres ${percent("0")}% | Négatifs ${percent("-1")}%`;
    render();
    const latency = lastStats.latency_ms ? `${lastStats.latency_ms.toFixed(0)} ms` : "OK";
    $("status").textContent = `Analyse terminée (${latency})`;
  } catch (error) {
    $("status").textContent = error.message;
  } finally {
    setLoading(false);
  }
};

$("copy").onclick = () =>
  navigator.clipboard.writeText(predictions.map((item) => `${item.sentiment}: ${item.text}`).join("\n"));
