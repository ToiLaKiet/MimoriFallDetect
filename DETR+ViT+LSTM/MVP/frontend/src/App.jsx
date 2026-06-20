import { useEffect, useMemo, useState } from "react";
import { getConfig, getHealth, processImages, reloadModels } from "./api.js";
import FrameVideoPlayer from "./FrameVideoPlayer.jsx";
import LiveCameraDemo from "./LiveCameraDemo.jsx";

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function buildPreviewUrls(files) {
  return [...files]
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((file) => URL.createObjectURL(file));
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [files, setFiles] = useState([]);
  const [folderPath, setFolderPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [mode, setMode] = useState("batch");

  const refreshStatus = async () => {
    setError("");
    try {
      const [healthData, configData] = await Promise.all([getHealth(), getConfig()]);
      setHealth(healthData);
      setConfig(configData);
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  useEffect(() => {
    return () => {
      previewUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [previewUrls]);

  const sortedFiles = useMemo(
    () => [...files].sort((a, b) => a.name.localeCompare(b.name)),
    [files],
  );

  const handleProcess = async () => {
    if (!folderPath.trim() && sortedFiles.length === 0) {
      setError("Upload images or provide a local folder path.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    previewUrls.forEach((url) => URL.revokeObjectURL(url));
    const urls = sortedFiles.length > 0 ? buildPreviewUrls(sortedFiles) : [];
    setPreviewUrls(urls);
    try {
      const data = await processImages(sortedFiles, folderPath);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReload = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await reloadModels();
      setHealth(data);
      await refreshStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Fall Detection MVP</h1>
        <p>
          Upload a sequence of frames. Backend detects the largest person, crops and
          extracts MMPose ViTPose embeddings, then runs LSTM when the sliding window
          reaches 10 frames.
        </p>
      </header>

      <section className="panel">
        <h2>Backend status</h2>
        <dl className="status-grid">
          <div>
            <dt>Health</dt>
            <dd>{health?.ok ? (health?.models_loaded ? "Models loaded" : "Ready (models not loaded)") : "Not ready"}</dd>
          </div>
          <div>
            <dt>Device</dt>
            <dd>{health?.device || "—"}</dd>
          </div>
          <div>
            <dt>LSTM checkpoint</dt>
            <dd>{config?.checkpoint || "—"}</dd>
          </div>
          <div>
            <dt>MMPose config</dt>
            <dd>{config?.mmpose_config || "—"}</dd>
          </div>
          <div>
            <dt>MMPose checkpoint</dt>
            <dd>{config?.mmpose_checkpoint || "—"}</dd>
          </div>
          <div>
            <dt>Window size</dt>
            <dd>{config?.window_size ?? "—"}</dd>
          </div>
          <div>
            <dt>Checkpoint epoch</dt>
            <dd>{health?.checkpoint_meta?.epoch ?? "—"}</dd>
          </div>
          <div>
            <dt>Val accuracy</dt>
            <dd>
              {health?.checkpoint_meta?.val_acc != null
                ? formatPercent(health.checkpoint_meta.val_acc)
                : "—"}
            </dd>
          </div>
        </dl>
        <div className="actions">
          <button type="button" className="secondary" onClick={refreshStatus} disabled={loading}>
            Refresh status
          </button>
          <button type="button" className="secondary" onClick={handleReload} disabled={loading}>
            Reload models
          </button>
        </div>
      </section>

      {/* <div className="mode-tabs">
        <button
          type="button"
          className={mode === "batch" ? "tab active" : "tab"}
          onClick={() => setMode("batch")}
        >
          Batch upload
        </button>
        <button
          type="button"
          className={mode === "live" ? "tab active" : "tab"}
          onClick={() => setMode("live")}
        >
          Live camera
        </button>
      </div> */}

      {mode === "batch" ? (
      <section className="panel">
        <h2>Process image sequence</h2>
        <div className="upload-row">
          <input
            type="file"
            accept="image/*,.zip"
            multiple
            onChange={(event) => setFiles(Array.from(event.target.files || []))}
          />
          <input
            type="text"
            placeholder="Optional local folder path on backend machine"
            value={folderPath}
            onChange={(event) => setFolderPath(event.target.value)}
          />
          <div className="muted">
            {sortedFiles.length > 0
              ? `${sortedFiles.length} file(s) selected`
              : "Or provide a folder path readable by the Flask server."}
          </div>
        </div>
        <div className="actions">
          <button type="button" onClick={handleProcess} disabled={loading}>
            {loading ? "Processing..." : "Run pipeline"}
          </button>
        </div>
      </section>
      )
       : (
        <section className="panel">
          <h2>Live camera demo</h2>
          <LiveCameraDemo windowSize={config?.window_size ?? 10} />
        </section>
      )
      }

      {mode === "batch" && error ? <div className="error">{error}</div> : null}

      {mode === "batch" && result ? (
        <section className="panel">
          <h2>Results</h2>
          <div className="summary">
            <span>Total frames: {result.frames_total}</span>
            <span>Processed: {result.frames_processed}</span>
            <span>Predictions: {result.predictions?.length || 0}</span>
          </div>

          {result.predictions?.length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Frame #</th>
                    <th>Name</th>
                    <th>Label</th>
                    <th>Confidence</th>
                    <th>Normal</th>
                    <th>Fall</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.map((item) => (
                    <tr key={`${item.frame_index}-${item.frame_name}`}>
                      <td>{item.frame_index + 1}</td>
                      <td>{item.frame_name}</td>
                      <td className={item.label === "fall" ? "label-fall" : "label-normal"}>
                        {item.label}
                      </td>
                      <td>{formatPercent(item.confidence)}</td>
                      <td>{formatPercent(item.probabilities.normal)}</td>
                      <td>{formatPercent(item.probabilities.fall)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">No predictions yet. Need at least 10 valid frames.</p>
          )}

          {result.fall_confirmation ? (
            <div className="fall-confirmation">
              <h3 className="subsection-title">Fall confirmation (bbox stability)</h3>
              {result.fall_confirmation.last_fall_frame_index === null ? (
                <p className="muted">No fall prediction in this sequence.</p>
              ) : (
                <>
                  <div className="summary">
                    <span>
                      Last fall frame: #{result.fall_confirmation.last_fall_frame_index + 1}{" "}
                      ({result.fall_confirmation.last_fall_frame_name})
                    </span>
                    {result.fall_confirmation.stability ? (
                      <>
                        <span>
                          Stability window:{" "}
                          {result.fall_confirmation.stability.start_index + 1}–
                          {result.fall_confirmation.stability.end_index + 1} (
                          {result.fall_confirmation.stability.frames_available}/
                          {result.fall_confirmation.stability.frames_requested} frames)
                        </span>
                        <span>
                          Mean IoU:{" "}
                          {result.fall_confirmation.stability.mean_iou !== null
                            ? result.fall_confirmation.stability.mean_iou.toFixed(3)
                            : "—"}
                        </span>
                        <span>
                          Mean center shift:{" "}
                          {result.fall_confirmation.stability.mean_center_shift_ratio !== null
                            ? result.fall_confirmation.stability.mean_center_shift_ratio.toFixed(
                                3,
                              )
                            : "—"}
                        </span>
                      </>
                    ) : null}
                  </div>
                  <p
                    className={
                      result.fall_confirmation.trigger_agent
                        ? "confirm-alert"
                        : "confirm-muted"
                    }
                  >
                    {result.fall_confirmation.trigger_agent
                      ? "Confirmed fall — bbox stable after last fall frame (ready to trigger Agent)."
                      : result.fall_confirmation.stability?.reason ===
                          "bbox_changed_too_much"
                        ? "Fall detected but bbox still moving — not confirmed."
                        : result.fall_confirmation.stability?.reason ===
                            "not_enough_frames_after_last_fall"
                          ? "Fall detected but not enough frames after last fall for stability check."
                          : "Fall not confirmed."}
                  </p>
                </>
              )}
            </div>
          ) : null}

          <h3 className="subsection-title">Sequence playback</h3>
          {previewUrls.length > 0 ? (
            <FrameVideoPlayer frames={result.frames} imageUrls={previewUrls} />
          ) : (
            <p className="muted">
              Video playback is available when you upload images from the browser. For
              folder_path on the server, only the results table is shown.
            </p>
          )}
        </section>
      ) : null}
    </div>
  );
}
