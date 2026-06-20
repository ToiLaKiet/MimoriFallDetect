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

export function getConfig() {
  return fetchJson("/api/config");
}

export function reloadModels() {
  return fetchJson("/api/reload", { method: "POST" });
}

export function processImages(files, folderPath = "") {
  const formData = new FormData();
  for (const file of files) {
    formData.append("images", file);
  }
  if (folderPath.trim()) {
    formData.append("folder_path", folderPath.trim());
  }
  return fetchJson("/api/process", {
    method: "POST",
    body: formData,
  });
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
