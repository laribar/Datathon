// components/ui/EmotionCam.jsx
import React, { useEffect, useRef, useState } from "react";

const EmotionCam = ({
  backendUrl = "http://127.0.0.1:8000/api/emotion",
  intervalMs = 500,
  topN = 3,
  width = 640,
  height = 480,
  onResult,            // (resultJson) => void
  autoStart = true,    // inicia automaticamente
  jpegQuality = 0.6,   // reduz payload
  className = "",
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const drawRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [lastDominant, setLastDominant] = useState(null);

  useEffect(() => {
    let stopped = false;
    let timer = null;

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width, height },
          audio: false,
        });
        if (stopped) return;
        const v = videoRef.current;
        v.srcObject = stream;
        await v.play();

        setRunning(true);

        // primeiro tick imediato
        tick();
        timer = setInterval(tick, Math.max(100, intervalMs));
      } catch (e) {
        console.error("EmotionCam: erro ao iniciar webcam:", e);
      }
    }

    async function tick() {
      try {
        const v = videoRef.current;
        const c = canvasRef.current;
        if (!v || !c) return;

        const ctx = c.getContext("2d"); // compatível com mais browsers
        ctx.drawImage(v, 0, 0, c.width, c.height);

        // JPEG para reduzir tráfego
        const dataUrl = c.toDataURL("image/jpeg", jpegQuality);
        const base64 = dataUrl.split(",")[1];

        const resp = await fetch(backendUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_base64: base64,
            detect_faces: true,
            top_n: topN,
          }),
        });
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(`HTTP ${resp.status} ${txt}`);
        }
        const json = await resp.json();
        onResult?.(json);
        draw(json);
      } catch (e) {
        console.warn("EmotionCam tick erro:", e.message);
      }
    }

    function draw(result) {
      try {
        const overlay = drawRef.current;
        if (!overlay) return;

        const ctx = overlay.getContext("2d");
        // limpar overlay
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const faces = Array.isArray(result?.faces) ? result.faces : [];
        faces.forEach((f) => {
          const b = f?.box;
          if (b && b.length >= 4) {
            const [x, y, w, h] = b;
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#2F9FF8";
            ctx.strokeRect(x, y, w, h);

            const label = f?.dominant
              ? `${f.dominant.label} (${Math.round(f.dominant.score * 100)}%)`
              : "—";
            ctx.fillStyle = "#2F9FF8";
            ctx.font = "16px system-ui, Arial";
            ctx.fillText(label, x + 6, Math.max(16, y - 6));
          }
        });

        const dom = result?.dominant_overall;
        setLastDominant(dom ? `${dom.label} (${(dom.score * 100).toFixed(1)}%)` : "—");
      } catch (e) {
        console.warn("EmotionCam draw erro:", e.message);
      }
    }

    if (autoStart) start();

    return () => {
      stopped = true;
      setRunning(false);
      if (timer) clearInterval(timer);
      const v = videoRef.current;
      const stream = v?.srcObject;
      if (stream && stream.getTracks) stream.getTracks().forEach((t) => t.stop());
      if (v) v.srcObject = null;
    };
  }, [backendUrl, intervalMs, topN, width, height, jpegQuality, autoStart, onResult]);

  return (
    <div className={`relative inline-block ${className}`} style={{ width, height }}>
      {/* vídeo base */}
      <video
        ref={videoRef}
        width={width}
        height={height}
        autoPlay
        playsInline
        muted
        className="rounded-lg object-cover w-full h-full bg-black"
      />
      {/* canvas para enviar frame */}
      <canvas ref={canvasRef} width={width} height={height} className="hidden" />
      {/* overlay de desenho */}
      <canvas
        ref={drawRef}
        width={width}
        height={height}
        className="absolute inset-0 pointer-events-none"
        style={{ borderRadius: 12 }}
      />
      {/* badge de dominante */}
      <div className="absolute top-3 left-3 bg-black/60 text-white text-xs px-2 py-1 rounded">
        {lastDominant ?? "—"}
      </div>

      {/* status simples */}
      <div className="absolute top-3 right-3">
        <span
          className={`inline-block w-2 h-2 rounded-full ${
            running ? "bg-green-400" : "bg-red-500"
          }`}
          title={running ? "Analisando" : "Parado"}
        />
      </div>
    </div>
  );
};

export default EmotionCam;