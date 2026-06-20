import { useCallback, useEffect, useRef, useState } from "react";

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function drawFrame(canvas, image, frame, displayWidth, displayHeight) {
  const ctx = canvas.getContext("2d");
  if (!ctx || !image) return;

  const scale = Math.min(displayWidth / image.naturalWidth, displayHeight / image.naturalHeight);
  const drawW = image.naturalWidth * scale;
  const drawH = image.naturalHeight * scale;
  const offsetX = (displayWidth - drawW) / 2;
  const offsetY = (displayHeight - drawH) / 2;

  ctx.clearRect(0, 0, displayWidth, displayHeight);
  ctx.fillStyle = "#0a0e14";
  ctx.fillRect(0, 0, displayWidth, displayHeight);
  ctx.drawImage(image, offsetX, offsetY, drawW, drawH);

  if (frame?.bbox_xyxy) {
    const [x1, y1, x2, y2] = frame.bbox_xyxy;
    const sx = scale;
    const sy = scale;
    const bx = offsetX + x1 * sx;
    const by = offsetY + y1 * sy;
    const bw = (x2 - x1) * sx;
    const bh = (y2 - y1) * sy;

    const isFall = frame.prediction?.label === "fall";
    ctx.strokeStyle = frame.bbox_fallback ? "#fbbf24" : isFall ? "#f87171" : "#4ade80";
    ctx.lineWidth = 3;
    ctx.strokeRect(bx, by, bw, bh);

    const label = frame.prediction
      ? `${frame.prediction.label} ${formatPercent(frame.prediction.confidence)}`
      : frame.bbox_fallback
        ? "person (fallback)"
        : "person";
    ctx.font = "600 14px system-ui, sans-serif";
    const padding = 6;
    const textW = ctx.measureText(label).width;
    const tagH = 22;
    const tagY = Math.max(offsetY, by - tagH - 4);
    ctx.fillStyle = isFall ? "rgba(248,113,113,0.92)" : "rgba(74,222,128,0.92)";
    if (!frame.prediction) ctx.fillStyle = "rgba(251,191,36,0.92)";
    ctx.fillRect(bx, tagY, textW + padding * 2, tagH);
    ctx.fillStyle = "#0f1419";
    ctx.fillText(label, bx + padding, tagY + 16);
  }

  if (frame?.error) {
    ctx.fillStyle = "rgba(239,68,68,0.85)";
    ctx.fillRect(8, displayHeight - 36, displayWidth - 16, 28);
    ctx.fillStyle = "#fff";
    ctx.font = "12px system-ui";
    ctx.fillText(frame.error.slice(0, 80), 16, displayHeight - 18);
  }
}

export default function FrameVideoPlayer({ frames, imageUrls }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const imageCache = useRef(new Map());

  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(8);
  const [displaySize, setDisplaySize] = useState({ w: 960, h: 540 });

  const total = frames?.length || 0;
  const currentFrame = frames?.[currentIndex] ?? null;
  const imageUrl = imageUrls?.[currentIndex] ?? null;

  const loadImage = useCallback(
    (url) =>
      new Promise((resolve, reject) => {
        if (!url) {
          reject(new Error("no url"));
          return;
        }
        if (imageCache.current.has(url)) {
          resolve(imageCache.current.get(url));
          return;
        }
        const img = new Image();
        img.onload = () => {
          imageCache.current.set(url, img);
          resolve(img);
        };
        img.onerror = reject;
        img.src = url;
      }),
    [],
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;

    const observer = new ResizeObserver(([entry]) => {
      const width = Math.floor(entry.contentRect.width);
      const height = Math.max(280, Math.floor(width * 9 / 16));
      setDisplaySize({ w: width, h: height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return undefined;

    let cancelled = false;
    loadImage(imageUrl)
      .then((img) => {
        if (!cancelled) {
          drawFrame(canvas, img, currentFrame, displaySize.w, displaySize.h);
        }
      })
      .catch(() => {
        if (!cancelled && canvas) {
          const ctx = canvas.getContext("2d");
          ctx?.clearRect(0, 0, displaySize.w, displaySize.h);
          ctx.fillStyle = "#1e293b";
          ctx.fillRect(0, 0, displaySize.w, displaySize.h);
          ctx.fillStyle = "#94a3b8";
          ctx.font = "14px system-ui";
          ctx.fillText("Preview unavailable (use file upload)", 24, displaySize.h / 2);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [currentIndex, currentFrame, displaySize, imageUrl, loadImage]);

  useEffect(() => {
    if (!playing || total === 0) return undefined;
    const timer = setInterval(() => {
      setCurrentIndex((idx) => (idx + 1 >= total ? 0 : idx + 1));
    }, 1000 / fps);
    return () => clearInterval(timer);
  }, [playing, fps, total]);

  useEffect(() => {
    setCurrentIndex(0);
    setPlaying(false);
  }, [frames]);

  if (!frames?.length) return null;

  return (
    <div className="video-player">
      <div className="video-stage" ref={containerRef}>
        <canvas
          ref={canvasRef}
          width={displaySize.w}
          height={displaySize.h}
          className="video-canvas"
        />
        <div className="video-overlay-bar">
          <span>
            Frame {currentIndex + 1} / {total}
            {currentFrame?.name ? ` · ${currentFrame.name}` : ""}
          </span>
          {currentFrame?.prediction ? (
            <span
              className={
                currentFrame.prediction.label === "fall" ? "label-fall" : "label-normal"
              }
            >
              {currentFrame.prediction.label}{" "}
              {formatPercent(currentFrame.prediction.confidence)}
            </span>
          ) : (
            <span className="muted">
              Buffer {currentFrame?.buffer_size ?? 0}/10
              {currentFrame?.buffer_size >= 10 ? "" : " (warming up)"}
            </span>
          )}
        </div>
      </div>

      <div className="video-controls">
        <button
          type="button"
          className="secondary"
          onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
          disabled={currentIndex === 0}
        >
          Prev
        </button>
        <button type="button" onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => setCurrentIndex((i) => Math.min(total - 1, i + 1))}
          disabled={currentIndex >= total - 1}
        >
          Next
        </button>
        <label className="fps-control">
          FPS
          <input
            type="range"
            min={1}
            max={30}
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
          />
          <span>{fps}</span>
        </label>
      </div>

      <input
        type="range"
        className="video-scrubber"
        min={0}
        max={Math.max(0, total - 1)}
        value={currentIndex}
        onChange={(e) => {
          setPlaying(false);
          setCurrentIndex(Number(e.target.value));
        }}
      />

      {imageUrls?.length > 0 ? (
        <div className="filmstrip">
          {frames.map((frame, idx) => (
            <button
              key={`${frame.index}-${frame.name}`}
              type="button"
              className={`filmstrip-item${idx === currentIndex ? " active" : ""}`}
              onClick={() => {
                setPlaying(false);
                setCurrentIndex(idx);
              }}
              title={frame.name}
            >
              <img src={imageUrls[idx]} alt={frame.name} />
              <span>{idx + 1}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
