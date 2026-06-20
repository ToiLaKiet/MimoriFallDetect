function formatPercent(value) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatSeconds(value) {
  if (value == null) return "—";
  return `${value.toFixed(1)}s`;
}

const STATE_LABELS = {
  idle: "Standby",
  falling: "Fall detected",
  monitoring: "Monitoring immobility",
  triggered: "Alert sent",
  cooldown: "Cooldown",
};

export default function AgentPanel({ alert, frame, windowSize }) {
  const prediction = frame?.prediction;
  const label = prediction?.label ?? "—";
  const state = alert?.state ?? "idle";

  return (
    <div className="card agent-panel">
      <h2>Agent Status</h2>

      <div className="agent-row">
        <span className="agent-label">Prediction</span>
        <span className={`agent-value ${label === "fall" ? "fall" : label === "normal" ? "normal" : ""}`}>
          {label !== "—" ? `${label} (${formatPercent(prediction?.confidence)})` : "—"}
        </span>
      </div>

      <div className="agent-row">
        <span className="agent-label">Buffer</span>
        <span className="agent-value">
          {frame?.buffer_size ?? 0}/{windowSize}
        </span>
      </div>

      <div className="agent-row">
        <span className="agent-label">Alert state</span>
        <span className={`agent-value ${state === "monitoring" ? "monitoring" : ""}`}>
          {STATE_LABELS[state] ?? state}
        </span>
      </div>

      {state === "monitoring" ? (
        <>
          <div className="agent-row">
            <span className="agent-label">Timer</span>
            <span className="agent-value">
              {formatSeconds(alert.monitoring_elapsed_s)} / {formatSeconds(alert.stability_seconds)}
            </span>
          </div>
          <div className="agent-row">
            <span className="agent-label">Bbox stable</span>
            <span className={`agent-value ${alert.bbox_stable ? "normal" : "fall"}`}>
              {alert.bbox_stable == null ? "—" : alert.bbox_stable ? "Yes" : "No"}
            </span>
          </div>
        </>
      ) : null}

      {state === "cooldown" && alert.cooldown_remaining_s != null ? (
        <div className="agent-row">
          <span className="agent-label">Cooldown</span>
          <span className="agent-value">{formatSeconds(alert.cooldown_remaining_s)}</span>
        </div>
      ) : null}

      <div className="agent-row">
        <span className="agent-label">Agent</span>
        <span className="agent-value">
          {alert?.trigger_agent || alert?.agent_result?.stub
            ? "Triggered (stub)"
            : "Standby"}
        </span>
      </div>

      {alert?.reason ? (
        <p className="muted" style={{ marginTop: "0.75rem" }}>
          {alert.reason.replace(/_/g, " ")}
        </p>
      ) : null}

      {alert?.trigger_agent ? (
        <div className="alert-banner">Fall confirmed — agent notified</div>
      ) : null}
    </div>
  );
}
