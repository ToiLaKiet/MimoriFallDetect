import { useCallback, useEffect, useRef, useState } from "react";
import { liveSendFrames, liveStart, liveStop } from "./api.js";

// UPFall training: 10 frames span ~0.52s → ~19fps. Target ~18fps for LSTM window.
const CAPTURE_FPS = 30;
const INFERENCE_TARGET_FPS = 18;
const CAPTURE_INTERVAL_MS = 1000 / CAPTURE_FPS;
const INFERENCE_INTERVAL_MS = 1000 / INFERENCE_TARGET_FPS;
const MAX_QUEUE_SIZE = 45;
const QUEUE_TRIM_TO = 24;
const MAX_BATCH_SIZE = 8;

function formatPercent(value) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

export default function LiveView({ windowSize, onFrame, onAlert, onActiveChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);

  const frameQueueRef = useRef([]);
  const lastCaptureAtRef = useRef(0);
  const lastEnqueueAtRef = useRef(0);
  const lastProcessedCaptureAtRef = useRef(null);
  const captureLoopIdRef = useRef(null);
  const inferenceCountRef = useRef(0);
  const inferenceWindowStartRef = useRef(0);

  const [active, setActive] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [framesTotal, setFramesTotal] = useState(0);
  const [queueDepth, setQueueDepth] = useState(0);
  const [inferenceFps, setInferenceFps] = useState(null);

  const snapshotFrame = useCallback(() => {
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
      capturedAt: performance.now(),
    };
  }, []);

  const trimQueue = useCallback(() => {
    const queue = frameQueueRef.current;
    if (queue.length <= MAX_QUEUE_SIZE) return;
    frameQueueRef.current = queue.slice(queue.length - QUEUE_TRIM_TO);
    setQueueDepth(frameQueueRef.current.length);
  }, []);

  const enqueueForInference = useCallback(
    (snapshot) => {
      frameQueueRef.current.push(snapshot);
      trimQueue();
      setQueueDepth(frameQueueRef.current.length);
    },
    [trimQueue],
  );

  const pickNextQueuedFrame = useCallback(() => {
    const queue = frameQueueRef.current;
    if (queue.length === 0) return null;

    const lastProcessed = lastProcessedCaptureAtRef.current;
    if (lastProcessed == null) {
      const next = queue.shift();
      setQueueDepth(queue.length);
      return next;
    }

    const minAt = lastProcessed + INFERENCE_INTERVAL_MS * 0.85;
    let pickIndex = -1;
    for (let i = 0; i < queue.length; i += 1) {
      if (queue[i].capturedAt >= minAt) {
        pickIndex = i;
        break;
      }
    }

    if (pickIndex === -1) return null;

    const skipped = queue.splice(0, pickIndex);
    const next = queue.shift();
    setQueueDepth(queue.length);

    if (skipped.length > 0 && queue.length > MAX_QUEUE_SIZE / 2) {
      trimQueue();
    }

    return next;
  }, [trimQueue]);

  const captureLoop = useCallback(() => {
    if (!runningRef.current) return;

    const now = performance.now();
    if (now - lastCaptureAtRef.current >= CAPTURE_INTERVAL_MS) {
      lastCaptureAtRef.current = now;
      const snapshot = snapshotFrame();
      if (snapshot && now - lastEnqueueAtRef.current >= INFERENCE_INTERVAL_MS * 0.9) {
        lastEnqueueAtRef.current = snapshot.capturedAt;
        enqueueForInference(snapshot);
      }
    }

    captureLoopIdRef.current = requestAnimationFrame(captureLoop);
  }, [enqueueForInference, snapshotFrame]);

  const recordInferenceFps = useCallback((processedCount = 1) => {
    inferenceCountRef.current += processedCount;
    const now = performance.now();
    if (inferenceWindowStartRef.current === 0) {
      inferenceWindowStartRef.current = now;
      return;
    }
    const elapsed = (now - inferenceWindowStartRef.current) / 1000;
    if (elapsed >= 2) {
      setInferenceFps(inferenceCountRef.current / elapsed);
      inferenceCountRef.current = 0;
      inferenceWindowStartRef.current = now;
    }
  }, []);

  const collectInferenceBatch = useCallback(() => {
    const batch = [];
    while (batch.length < MAX_BATCH_SIZE) {
      const frame = pickNextQueuedFrame();
      if (!frame) break;
      batch.push(frame);
    }
    return batch;
  }, [pickNextQueuedFrame]);

  const inferenceLoop = useCallback(async () => {
    while (runningRef.current) {
      const batch = collectInferenceBatch();
      if (batch.length === 0) {
        await sleep(8);
        continue;
      }

      setProcessing(true);
      try {
        const result = await liveSendFrames(batch.map((item) => item.dataUrl));
        const lastFrame = batch[batch.length - 1];
        lastProcessedCaptureAtRef.current = lastFrame.capturedAt;
        recordInferenceFps(batch.length);
        setFramesTotal(result.frames_total);
        onFrame?.(result.frame);
        onAlert?.(result.alert);
        drawOverlay(canvasRef.current, videoRef.current, result.frame, {
          width: lastFrame.width,
          height: lastFrame.height,
        });
      } catch (err) {
        setError(err.message);
        runningRef.current = false;
        setActive(false);
        onActiveChange?.(false);
        break;
      } finally {
        setProcessing(false);
      }
    }
  }, [collectInferenceBatch, onActiveChange, onAlert, onFrame, recordInferenceFps]);

  const stopCamera = useCallback(async () => {
    runningRef.current = false;
    if (captureLoopIdRef.current != null) {
      cancelAnimationFrame(captureLoopIdRef.current);
      captureLoopIdRef.current = null;
    }
    frameQueueRef.current = [];
    lastProcessedCaptureAtRef.current = null;
    lastCaptureAtRef.current = 0;
    lastEnqueueAtRef.current = 0;
    inferenceCountRef.current = 0;
    inferenceWindowStartRef.current = 0;
    setQueueDepth(0);
    setInferenceFps(null);

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
      // ignore
    }
    setActive(false);
    setProcessing(false);
    onActiveChange?.(false);
  }, [onActiveChange]);

  const handleStart = async () => {
    setError("");
    onFrame?.(null);
    onAlert?.(null);
    setFramesTotal(0);
    setQueueDepth(0);
    setInferenceFps(null);
    try {
      await liveStart();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      runningRef.current = true;
      setActive(true);
      onActiveChange?.(true);
      captureLoop();
      inferenceLoop();
    } catch (err) {
      setError(err.message || "Cannot access camera.");
      await stopCamera();
    }
  };

  useEffect(() => {
    return () => {
      runningRef.current = false;
      if (captureLoopIdRef.current != null) {
        cancelAnimationFrame(captureLoopIdRef.current);
      }
      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop();
        }
      }
    };
  }, []);

  return (
    <div className="card">
      <div className="live-stage">
        <video ref={videoRef} className="live-video" playsInline muted />
        <canvas ref={canvasRef} className="live-overlay" />
        <canvas ref={captureRef} hidden />
      </div>

      <div className="live-stats">
        <span>Inferred: {framesTotal}</span>
        <span>Queue: {queueDepth}</span>
        <span>Target: {INFERENCE_TARGET_FPS} fps</span>
        <span>Infer: {inferenceFps != null ? `${inferenceFps.toFixed(1)} fps` : "—"}</span>
        <span>{processing ? "Processing…" : active ? "Live" : "Ready"}</span>
      </div>

      <div className="actions">
        {!active ? (
          <button type="button" onClick={handleStart}>
            Start camera
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
