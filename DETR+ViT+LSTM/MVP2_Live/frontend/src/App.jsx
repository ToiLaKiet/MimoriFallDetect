import { useState } from "react";
import AgentPanel from "./AgentPanel.jsx";
import LiveView from "./LiveView.jsx";

const WINDOW_SIZE = 10;

export default function App() {
  const [active, setActive] = useState(false);
  const [frame, setFrame] = useState(null);
  const [alert, setAlert] = useState(null);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>MimamoriFall</h1>
          <p>見守り — Live Fall Watch</p>
        </div>
        <div className={`status-pill ${active ? "live" : "stopped"}`}>
          <span className="status-dot" />
          {active ? "Live" : "Stopped"}
        </div>
      </header>

      <div className="layout">
        <LiveView
          windowSize={WINDOW_SIZE}
          onFrame={setFrame}
          onAlert={setAlert}
          onActiveChange={setActive}
        />
        <AgentPanel alert={alert} frame={frame} windowSize={WINDOW_SIZE} />
      </div>

      <p className="muted">
        RT-DETR-X → ViTPose embedding → LSTM (window {WINDOW_SIZE}). Webcam captures at
        30 fps; frames are queued at ~18 fps to match training temporal spacing (~0.5 s
        per window). Alert fires after fall→normal + 5s bbox stability.
      </p>
    </div>
  );
}
