// components/ui/EmotionCam.jsx
import React, { useEffect, useRef, useState } from "react";

const EmotionCam = ({
  backendUrl = "http://127.0.0.1:8000/api/emotion",
  intervalMs = 500,
  topN = 3,
  width = 640,
  height = 480,
  onResult,
  autoStart = true,
  jpegQuality = 0.6,
  className = "",
  mediaStream = null,   // stream externo (opcional)
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const drawRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [lastDominant, setLastDominant] = useState(null);

  // controla propriedade do stream (se fomos nós que criamos)
  const ownsStreamRef = useRef(false);

  useEffect(() => {
    let stopped = false;
    let timer = null;

    async function startLocalCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width, height },
          audio: false,
        });
        if (stopped) return;
        ownsStreamRef.current = true;
        await attachStream(stream);
      } catch (e) {
        console.error("EmotionCam: erro ao iniciar webcam:", e);
      }
    }

    async function attachStream(stream) {
      const v = videoRef.current;
      if (!v) return;

      try {
        v.srcObject = stream;

        // aguarda metadados para poder dar play com segurança
        await new Promise((resolve) => {
          if (v.readyState >= 1) return resolve();
          const onLoaded = () => {
            v.removeEventListener("loadedmetadata", onLoaded);
            resolve();
          };
          v.addEventListener("loadedmetadata", onLoaded, { once: true });
        });

        await v.play().catch(() => {});

        setRunning(true);

        // roda 1º tick e agenda próximos
        tick();
        timer = setInterval(tick, Math.max(100, intervalMs));
      } catch (e) {
        console.warn("EmotionCam attachStream erro:", e.message);
      }
    }

    async function tick() {
      try {
        const v = videoRef.current;
        const c = canvasRef.current;
        if (!v || !c) return;

        const ctx = c.getContext("2d");
        ctx.drawImage(v, 0, 0, c.width, c.height);

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
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json = await resp.json();
        onResult?.(json);
        draw(json);
      } catch (e) {
        if (!stopped) console.warn("EmotionCam tick erro:", e.message);
      }
    }

    function draw(result) {
      try {
        const overlay = drawRef.current;
        if (!overlay) return;
        const ctx = overlay.getContext("2d");
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

    // usa stream externo se vier; senão, abre local (se autoStart)
    if (mediaStream) {
      ownsStreamRef.current = false;
      attachStream(mediaStream);
    } else if (autoStart) {
      startLocalCamera();
    }

    return () => {
      stopped = true;
      setRunning(false);
      if (timer) clearInterval(timer);

      const v = videoRef.current;
      const s = v?.srcObject;

      // só paramos tracks se FOMOS nós que criamos o stream
      if (ownsStreamRef.current && s && s.getTracks) {
        try {
          s.getTracks().forEach((t) => t.stop());
        } catch {}
      }

      if (v) v.srcObject = null;
    };
  }, [backendUrl, intervalMs, topN, width, height, jpegQuality, autoStart, onResult, mediaStream]);

  return (
    <div className={`relative inline-block ${className}`} style={{ width, height }}>
      <video
        ref={videoRef}
        width={width}
        height={height}
        autoPlay
        playsInline
        muted
        className="rounded-lg object-cover w-full h-full bg-black"
      />
      <canvas ref={canvasRef} width={width} height={height} className="hidden" />
      <canvas
        ref={drawRef}
        width={width}
        height={height}
        className="absolute inset-0 pointer-events-none"
        style={{ borderRadius: 12 }}
      />
      <div className="absolute top-3 left-3 bg-black/60 text-white text-xs px-2 py-1 rounded">
        {lastDominant ?? "—"}
      </div>
      <div className="absolute top-3 right-3">
        <span
          className={`inline-block w-2 h-2 rounded-full ${running ? "bg-green-400" : "bg-red-500"}`}
          title={running ? "Analisando" : "Parado"}
        />
      </div>
    </div>
  );
};

export default EmotionCam;
  