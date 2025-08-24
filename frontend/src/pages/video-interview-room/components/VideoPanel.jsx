// pages/interview/components/VideoPanel.jsx
import React, { useState, useRef, useEffect, useCallback } from "react";
import Icon from "../../../components/AppIcon";
import Button from "../../../components/ui/Button";
import EmotionCam from "../../../components/ui/EmotionCam";

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

  // Dispositivos
  const [videoDevices, setVideoDevices] = useState([]);   // videoinput
  const [audioDevices, setAudioDevices] = useState([]);   // audioinput
  const [selectedCamId, setSelectedCamId] = useState("");
  const [selectedMicId, setSelectedMicId] = useState("");

  // Streams
  const [stream, setStream] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // Helpers
  const stopStream = useCallback(() => {
    try {
      stream?.getTracks()?.forEach((t) => t.stop());
    } catch {}
  }, [stream]);

  const refreshDevices = useCallback(async () => {
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all.filter((d) => d.kind === "videoinput");
      const mics = all.filter((d) => d.kind === "audioinput");
      setVideoDevices(cams);
      setAudioDevices(mics);
      if (!selectedCamId && cams[0]) setSelectedCamId(cams[0].deviceId);
      if (!selectedMicId && mics[0]) setSelectedMicId(mics[0].deviceId);
    } catch (e) {
      console.error(e);
      setErrorMsg("Não foi possível listar dispositivos. Verifique permissões.");
    }
  }, [selectedCamId, selectedMicId]);

  const startWithDevices = useCallback(
    async (camId, micId) => {
      setLoading(true);
      setErrorMsg("");
      try {
        // encerra stream anterior
        stopStream();

        const constraints = {
          video: isVideoOff
            ? false
            : camId
            ? { deviceId: { exact: camId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: micId ? { deviceId: { exact: micId } } : true, // se não houver, pega padrão
        };

        const newStream = await navigator.mediaDevices.getUserMedia(constraints);

        // aplica mute conforme prop
        newStream.getAudioTracks().forEach((t) => (t.enabled = !isAudioMuted));

        setStream(newStream);
      } catch (e) {
        console.error(e);
        setErrorMsg(
          "Falha ao iniciar câmera/microfone. Dê permissão no navegador e verifique se não estão em uso por outro app."
        );
      } finally {
        setLoading(false);
      }
    },
    [isVideoOff, isAudioMuted, stopStream]
  );

  // Init: pega permissão uma vez para liberar labels e popula devices
  useEffect(() => {
    (async () => {
      try {
        await startWithDevices("", ""); // padrão -> libera labels na maioria dos browsers
        await refreshDevices();
      } catch (e) {
        // ok
      }
    })();

    const onDeviceChange = () => refreshDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", onDeviceChange);

    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", onDeviceChange);
      stopStream();
    };
  }, [refreshDevices, startWithDevices, stopStream]);

  // Troca de câmera/microfone pelo usuário
  useEffect(() => {
    if (selectedCamId || selectedMicId) {
      startWithDevices(selectedCamId, selectedMicId);
    }
  }, [selectedCamId, selectedMicId, startWithDevices]);

  // Responder a mudanças externas de mute
  useEffect(() => {
    stream?.getAudioTracks()?.forEach((t) => (t.enabled = !isAudioMuted));
  }, [isAudioMuted, stream]);

  // Fullscreen
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

  // UI helpers
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

  const EMOTION_URL =
    import.meta?.env?.VITE_EMOTION_URL || "http://127.0.0.1:8000/api/emotion";

  return (
    <div
      ref={containerRef}
      className={`relative bg-gray-900 rounded-lg overflow-hidden shadow-elevation-2 ${
        isMainVideo ? "h-full" : "h-48"
      }`}
    >
      {/* Topbar: seletor de câmera e microfone */}
      <div className="absolute z-20 top-3 left-3 flex flex-wrap items-center gap-2 bg-black/40 backdrop-blur-sm px-2 py-1 rounded">
        {/* Câmera */}
        <div className="flex items-center gap-2">
          <Icon name="Camera" size={14} className="text-white/80" />
          <select
            className="text-xs bg-transparent text-white/90 border border-white/20 rounded px-2 py-1 outline-none"
            value={selectedCamId}
            onChange={(e) => setSelectedCamId(e.target.value)}
            title="Selecionar câmera"
          >
            {videoDevices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Câmera ${d.deviceId.slice(0, 6)}`}
              </option>
            ))}
          </select>
        </div>

        {/* Microfone */}
        <div className="flex items-center gap-2">
          <Icon name="Mic" size={14} className="text-white/80" />
          <select
            className="text-xs bg-transparent text-white/90 border border-white/20 rounded px-2 py-1 outline-none"
            value={selectedMicId}
            onChange={(e) => setSelectedMicId(e.target.value)}
            title="Selecionar microfone"
          >
            {audioDevices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Microfone ${d.deviceId.slice(0, 6)}`}
              </option>
            ))}
          </select>
        </div>

        <Button
          variant="ghost"
          size="sm"
          className="text-white/80 hover:bg-white/10"
          onClick={refreshDevices}
          title="Atualizar lista de dispositivos"
        >
          <Icon name="RefreshCw" size={14} />
        </Button>
      </div>

      {/* Área de vídeo / placeholder */}
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
            {/* EmotionCam usa o stream (ele pode conter áudio; o componente ignora se não usar) */}
            <EmotionCam
              className="w-full h-full"
              width={1280}
              height={720}
              backendUrl={EMOTION_URL}
              intervalMs={500}
              topN={3}
              mediaStream={stream}
              onResult={() => {}}
            />

            {/* Overlays */}
            {loading && (
              <div className="absolute inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-10">
                <div className="flex items-center gap-2 text-white/90 text-sm">
                  <Icon name="Loader2" size={16} className="animate-spin" />
                  Iniciando dispositivos…
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
          <div className="absolute bottom-3 left-3 bg-red-600 rounded-full p-2" title="Microfone mutado">
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

        {/* Nome */}
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

      {/* Elemento <audio> local (muted) só para manter a track ativa sem eco.
          Se você quiser monitorar o áudio localmente, remova 'muted' (não recomendado). */}
      <audio
        autoPlay
        playsInline
        muted
        ref={(el) => {
          if (el && stream) el.srcObject = stream;
        }}
        style={{ display: "none" }}
      />
    </div>
  );
};

export default VideoPanel;
