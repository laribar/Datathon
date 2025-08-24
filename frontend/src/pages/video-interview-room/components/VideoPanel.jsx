// pages/interview/components/VideoPanel.jsx
import React, { useState, useRef, useEffect, useCallback } from "react";
import Icon from "../../../components/AppIcon";
import Button from "../../../components/ui/Button";
import EmotionCam from "../../../components/ui/EmotionCam";

// ✅ Painel de vídeo com seletor de câmera profissional
// - Lista e seleciona webcams (videoinput) via mediaDevices
// - Hot-swap (devicechange) quando usuário pluga/remove a câmera
// - Passa o MediaStream para o EmotionCam via prop `mediaStream`
// - Fallback seguro e mensagens claras de erro

const VideoPanel = ({
  isMainVideo = false,
  participantName = "Participante",
  isAudioMuted = false,
  isVideoOff = false,
  connectionQuality = "good",
  onToggleAudio,
  onToggleVideo,
  onToggleScreenShare,
  isScreenSharing = false,
}) => {
  const containerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // ---- Estado do seletor de câmera ----
  const [devices, setDevices] = useState([]); // [{deviceId,label,kind}]
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [stream, setStream] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // ---- Utils ----
  const stopStream = useCallback(() => {
    try {
      stream?.getTracks()?.forEach((t) => t.stop());
    } catch {}
  }, [stream]);

  const refreshDevices = useCallback(async () => {
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      const cams = list.filter((d) => d.kind === "videoinput");
      setDevices(cams);
      // Se ainda não temos seleção, pega a primeira
      if (!selectedDeviceId && cams[0]) setSelectedDeviceId(cams[0].deviceId);
    } catch (e) {
      console.error(e);
      setErrorMsg("Não foi possível listar as câmeras. Verifique permissões.");
    }
  }, [selectedDeviceId]);

  const startWith = useCallback(
    async (deviceId) => {
      setLoading(true);
      setErrorMsg("");
      try {
        // encerra stream anterior
        stopStream();

        const constraints = {
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };

        const newStream = await navigator.mediaDevices.getUserMedia(constraints);
        setStream(newStream);
      } catch (e) {
        console.error(e);
        setErrorMsg(
          "Falha ao iniciar a câmera. Garanta que o browser tem permissão e que a câmera não está em uso por outro app."
        );
      } finally {
        setLoading(false);
      }
    },
    [stopStream]
  );

  // ---- Init: pede permissão uma vez e popula devices (labels) ----
  useEffect(() => {
    (async () => {
      // Em muitos browsers os labels só aparecem após permissão
      try {
        if (!isVideoOff) {
          await startWith(""); // pega padrão para liberar labels
        }
        await refreshDevices();
      } catch (e) {
        console.error(e);
      }
    })();

    // hot-swap de dispositivos
    const onDeviceChange = () => refreshDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", onDeviceChange);

    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", onDeviceChange);
      stopStream();
    };
  }, [isVideoOff, refreshDevices, startWith, stopStream]);

  // ---- Quando usuário troca de câmera ----
  useEffect(() => {
    if (selectedDeviceId) startWith(selectedDeviceId);
  }, [selectedDeviceId, startWith]);

  // ---- Fullscreen ----
  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement && containerRef.current) {
        await containerRef.current.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // ---- UI helpers ----
  const getQualityColor = () => {
    switch (connectionQuality) {
      case "excellent":
        return "text-green-400";
      case "good":
        return "text-green-500";
      case "fair":
        return "text-yellow-500";
      case "poor":
        return "text-red-500";
      default:
        return "text-gray-400";
    }
  };

  const getQualityBars = () => {
    const bars = [];
    const levels = { excellent: 4, good: 3, fair: 2, poor: 1 };
    const level = levels?.[connectionQuality] || 0;

    for (let i = 0; i < 4; i++) {
      bars.push(
        <div
          key={i}
          className={`w-1 bg-current transition-all duration-300 ${
            i < level ? "opacity-100" : "opacity-30"
          } ${i === 0 ? "h-2" : i === 1 ? "h-3" : i === 2 ? "h-4" : "h-5"}`}
        />
      );
    }
    return bars;
  };

  // ---- URL do backend de emoções (Vite) ----
  const EMOTION_URL =
    import.meta?.env?.VITE_EMOTION_URL || "http://127.0.0.1:8000/api/emotion";

  return (
    <div
      ref={containerRef}
      className={`relative bg-gray-900 rounded-lg overflow-hidden shadow-elevation-2 ${
        isMainVideo ? "h-full" : "h-48"
      }`}
    >
      {/* Topbar: seletor de câmera */}
      {!isVideoOff && (
        <div className="absolute z-20 top-3 left-3 flex items-center gap-2 bg-black/40 backdrop-blur-sm px-2 py-1 rounded">
          <Icon name="Camera" size={14} className="text-white/80" />
          <select
            className="text-xs bg-transparent text-white/90 border border-white/20 rounded px-2 py-1 outline-none"
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            title="Selecionar câmera"
          >
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Câmera ${d.deviceId.slice(0, 6)}`}
              </option>
            ))}
          </select>

          <Button
            variant="ghost"
            size="sm"
            className="text-white/80 hover:bg-white/10"
            onClick={refreshDevices}
            title="Atualizar lista de câmeras"
          >
            <Icon name="RefreshCw" size={14} />
          </Button>
        </div>
      )}

      {/* Video / Placeholder */}
      <div className="relative w-full h-full">
        {isVideoOff ? (
          <div className="flex items-center justify-center h-full bg-gray-800">
            <div className="text-center">
              <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-3">
                <Icon name="User" size={32} className="text-gray-400" />
              </div>
              <p className="text-gray-300 font-medium">{participantName}</p>
              <p className="text-gray-500 text-sm mt-1">Câmera desligada</p>
            </div>
          </div>
        ) : (
          <>
            {/* EmotionCam recebe o stream selecionado */}
            <EmotionCam
              className="w-full h-full"
              width={1280}
              height={720}
              backendUrl={EMOTION_URL}
              intervalMs={500}
              topN={3}
              mediaStream={stream} // <- chave: usa a câmera escolhida
              onResult={(json) => {
                // console.log("Emotion:", json?.dominant_overall);
              }}
            />

            {/* Loading / Erro overlay */}
            {loading && (
              <div className="absolute inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-10">
                <div className="flex items-center gap-2 text-white/90 text-sm">
                  <Icon name="Loader2" size={16} className="animate-spin" />
                  Iniciando câmera…
                </div>
              </div>
            )}
            {errorMsg && (
              <div className="absolute bottom-3 left-3 right-3 bg-red-600/90 text-white text-xs px-3 py-2 rounded z-10">
                {errorMsg}
              </div>
            )}
          </>
        )}

        {/* Indicador de qualidade */}
        <div className={`absolute top-3 right-3 flex items-center gap-1 ${getQualityColor()}`}>
          <div className="flex items-end gap-0.5 h-5">{getQualityBars()}</div>
        </div>

        {/* Mic mutado */}
        {isAudioMuted && (
          <div className="absolute bottom-3 left-3 bg-red-600 rounded-full p-2">
            <Icon name="MicOff" size={16} className="text-white" />
          </div>
        )}

        {/* Compartilhamento de tela */}
        {isScreenSharing && (
          <div className="absolute top-3 left-3 bg-blue-600 rounded-full px-3 py-1 flex items-center gap-2">
            <Icon name="Monitor" size={14} className="text-white" />
            <span className="text-white text-xs font-medium">Compartilhando tela</span>
          </div>
        )}

        {/* Nome do participante */}
        <div className="absolute bottom-3 left-3 bg-black/50 backdrop-blur-sm rounded px-3 py-1">
          <span className="text-white text-sm font-medium">{participantName}</span>
        </div>

        {/* Controles do vídeo principal */}
        {isMainVideo && (
          <div className="absolute bottom-4 right-4 flex gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="bg-black/50 backdrop-blur-sm hover:bg-black/70 text-white"
              onClick={toggleFullscreen}
              title={isFullscreen ? "Sair da tela cheia" : "Tela cheia"}
            >
              <Icon name={isFullscreen ? "Minimize2" : "Maximize2"} size={20} />
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoPanel;
