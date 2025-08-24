// components/ui/CameraSelector.jsx
import React, { useEffect, useRef, useState } from "react";

export default function CameraSelector({
  width = 1280,
  height = 720,
  onStreamChange,         // callback(stream) -> opcional (ex.: enviar para WebRTC ou EmotionCam)
  autoStart = true,       // inicia com a câmera padrão
}) {
  const videoRef = useRef(null);
  const [devices, setDevices] = useState([]);        // { deviceId, label, kind }
  const [videoDeviceId, setVideoDeviceId] = useState("");
  const [stream, setStream] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // lista câmeras (precisa de HTTPS e, em alguns browsers, permissão prévia p/ exibir labels)
  const refreshDevices = async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all.filter((d) => d.kind === "videoinput");
      setDevices(cams);
      if (!videoDeviceId && cams[0]) setVideoDeviceId(cams[0].deviceId);
    } catch (e) {
      setErrorMsg("Não foi possível listar dispositivos de vídeo.");
      console.error(e);
    }
  };

  const stopStream = () => {
    try {
      stream?.getTracks()?.forEach((t) => t.stop());
    } catch (_) {}
  };

  const startWith = async (deviceId) => {
    setLoading(true);
    setErrorMsg("");
    try {
      // pare qualquer stream anterior
      stopStream();

      const constraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: width }, height: { ideal: height } }
          : { facingMode: "user", width: { ideal: width }, height: { ideal: height } },
        audio: false,
      };

      const newStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(newStream);
      if (videoRef.current) videoRef.current.srcObject = newStream;
      onStreamChange?.(newStream);
      setLoading(false);
    } catch (e) {
      setLoading(false);
      setErrorMsg("Falha ao iniciar a câmera. Verifique permissões e se a câmera está em uso.");
      console.error(e);
    }
  };

  // inicialização
  useEffect(() => {
    (async () => {
      // em alguns browsers, para os labels aparecerem, é preciso pedir permissão ao menos uma vez
      if (autoStart) {
        await startWith("");
      }
      await refreshDevices();
    })();

    // hot‑swap de dispositivos
    const onChange = () => refreshDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", onChange);

    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", onChange);
      stopStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // quando o deviceId selecionado mudar, (re)inicia
  useEffect(() => {
    if (videoDeviceId) startWith(videoDeviceId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoDeviceId]);

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-2">
        <label className="text-sm">Câmera:</label>
        <select
          className="border rounded px-2 py-1 bg-transparent"
          value={videoDeviceId}
          onChange={(e) => setVideoDeviceId(e.target.value)}
        >
          {devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Câmera ${d.deviceId.slice(0, 6)}`}
            </option>
          ))}
        </select>
        {loading && <span className="text-xs opacity-70">Iniciando…</span>}
      </div>

      {errorMsg && (
        <div className="text-xs text-red-400 mb-2">{errorMsg}</div>
      )}

    <video
    autoPlay
    playsInline
    muted
    ref={(el) => {
        if (el && stream) {
        el.srcObject = stream;
        }
    }}
    className="w-full h-full"
    />
    </div>
  );
}
