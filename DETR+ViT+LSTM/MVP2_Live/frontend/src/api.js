const API_BASE = import.meta.env.VITE_API_BASE || "";

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

export function getHealth() {
  return fetchJson("/api/health");
}

export function liveStart() {
  return fetchJson("/api/live/start", { method: "POST" });
}

export function liveStop() {
  return fetchJson("/api/live/stop", { method: "POST" });
}

export function liveSendFrame(dataUrl) {
  return fetchJson("/api/live/frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl }),
  });
}

export function liveSendFrames(dataUrls) {
  if (dataUrls.length === 1) {
    return liveSendFrame(dataUrls[0]);
  }
  return fetchJson("/api/live/frames", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images: dataUrls }),
  });
}
