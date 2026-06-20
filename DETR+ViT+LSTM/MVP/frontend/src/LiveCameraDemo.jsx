import { useCallback, useEffect, useRef, useState } from "react";
import { liveSendFrame, liveStart, liveStop } from "./api.js";

function formatPercent(value) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function drawOverlay(canvas, video, frame, inferenceSize) {
  const ctx = canvas.getContext("2d");
  if (!ctx || !video) return;

  const width = video.videoWidth || video.clientWidth;
  const height = video.videoHeight || video.clientHeight;
  if (!width || !height) return;

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(video, 0, 0, width, height);

  if (!frame?.bbox_xyxy || !inferenceSize) return;

  const scaleX = width / inferenceSize.width;
  const scaleY = height / inferenceSize.height;

  const [x1, y1, x2, y2] = frame.bbox_xyxy;
  const bx = x1 * scaleX;
  const by = y1 * scaleY;
  const bw = (x2 - x1) * scaleX;
  const bh = (y2 - y1) * scaleY;

  const isFall = frame.prediction?.label === "fall";
  ctx.strokeStyle = frame.bbox_fallback ? "#fbbf24" : isFall ? "#f87171" : "#4ade80";
  ctx.lineWidth = 3;
  ctx.strokeRect(bx, by, bw, bh);

  const label = frame.prediction
    ? `${frame.prediction.label} ${formatPercent(frame.prediction.confidence)}`
    : `buffer ${frame.buffer_size}/10`;
  ctx.font = "600 16px system-ui, sans-serif";
  const padding = 8;
  const textW = ctx.measureText(label).width;
  const tagH = 26;
  const tagY = Math.max(0, by - tagH - 4);
  ctx.fillStyle = isFall ? "rgba(248,113,113,0.9)" : "rgba(74,222,128,0.9)";
  if (!frame.prediction) ctx.fillStyle = "rgba(30,41,59,0.9)";
  ctx.fillRect(bx, tagY, textW + padding * 2, tagH);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, bx + padding, tagY + 18);
}

export default function LiveCameraDemo({ windowSize = 10 }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);
  const busyRef = useRef(false);

  const [active, setActive] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [frameResult, setFrameResult] = useState(null);
  const [fallConfirmation, setFallConfirmation] = useState(null);
  const [framesTotal, setFramesTotal] = useState(0);

  const stopCamera = useCallback(async () => {
    runningRef.current = false;
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    try {
      await liveStop();
    } catch {
      // ignore stop errors when session was never started
    }
    setActive(false);
    setProcessing(false);
  }, []);

  const captureDataUrl = useCallback(() => {
    const video = videoRef.current;
    const capture = captureRef.current;
    if (!video || !capture || !video.videoWidth || !video.videoHeight) return null;

    const targetW = 640;
    const scale = targetW / video.videoWidth;
    const targetH = Math.round(video.videoHeight * scale);
    capture.width = targetW;
    capture.height = targetH;
    const ctx = capture.getContext("2d");
    ctx.drawImage(video, 0, 0, targetW, targetH);
    return {
      dataUrl: capture.toDataURL("image/jpeg", 0.75),
      width: targetW,
      height: targetH,
    };
  }, []);

  const processLoop = useCallback(async () => {
    while (runningRef.current) {
      if (busyRef.current) {
        await new Promise((resolve) => setTimeout(resolve, 50));
        continue;
      }
      const captured = captureDataUrl();
      if (!captured) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        continue;
      }

      busyRef.current = true;
      setProcessing(true);
      try {
        const result = await liveSendFrame(captured.dataUrl);
        setFrameResult(result.frame);
        setFallConfirmation(result.fall_confirmation);
        setFramesTotal(result.frames_total);
        drawOverlay(canvasRef.current, videoRef.current, result.frame, {
          width: captured.width,
          height: captured.height,
        });
      } catch (err) {
        setError(err.message);
        runningRef.current = false;
        setActive(false);
        break;
      } finally {
        busyRef.current = false;
        setProcessing(false);
      }
    }
  }, [captureDataUrl]);

  const handleStart = async () => {
    setError("");
    setFrameResult(null);
    setFallConfirmation(null);
    setFramesTotal(0);
    try {
      await liveStart();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      runningRef.current = true;
      setActive(true);
      processLoop();
    } catch (err) {
      setError(err.message || "Cannot access camera.");
      await stopCamera();
    }
  };

  useEffect(() => {
    return () => {
      runningRef.current = false;
      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop();
        }
      }
    };
  }, []);

  return (
    <div className="live-demo">
      <p className="muted">
        Webcam live demo: each captured frame is sent to the backend (RT-DETR → ViTPose → LSTM
        window {windowSize}). Processing runs sequentially — expect ~1–3s per frame on CPU.
      </p>

      <div className="live-stage">
        <video ref={videoRef} className="live-video" playsInline muted />
        <canvas ref={canvasRef} className="live-overlay" />
        <canvas ref={captureRef} className="live-capture" hidden />
      </div>

      <div className="live-stats">
        <span>Frames sent: {framesTotal}</span>
        <span>
          Buffer: {frameResult?.buffer_size ?? 0}/{windowSize}
        </span>
        <span>{processing ? "Processing…" : active ? "Live" : "Stopped"}</span>
      </div>

      {frameResult?.prediction ? (
        <div
          className={
            frameResult.prediction.label === "fall" ? "confirm-alert" : "confirm-muted"
          }
        >
          Current: {frameResult.prediction.label}{" "}
          {formatPercent(frameResult.prediction.confidence)}
        </div>
      ) : null}

      {fallConfirmation?.trigger_agent ? (
        <div className="confirm-alert">
          Confirmed fall — bbox stable (Agent trigger ready).
        </div>
      ) : null}

      {fallConfirmation?.stability?.reason === "not_enough_frames_after_last_fall" ? (
        <div className="confirm-muted">
          Fall detected — collecting {fallConfirmation.stability.frames_available}/
          {fallConfirmation.stability.frames_requested} frames for stability check…
        </div>
      ) : null}

      <div className="actions">
        {!active ? (
          <button type="button" onClick={handleStart}>
            Start live camera
          </button>
        ) : (
          <button type="button" className="secondary" onClick={stopCamera}>
            Stop
          </button>
        )}
      </div>

      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}
