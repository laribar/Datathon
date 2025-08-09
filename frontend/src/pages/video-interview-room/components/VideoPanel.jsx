import React, { useState, useRef, useEffect } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const VideoPanel = ({ 
  isMainVideo = false, 
  participantName = "Participante", 
  isAudioMuted = false, 
  isVideoOff = false,
  connectionQuality = "good",
  onToggleAudio,
  onToggleVideo,
  onToggleScreenShare,
  isScreenSharing = false
}) => {
  const videoRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    // Simulate video stream
    if (videoRef?.current && !isVideoOff) {
      // In real implementation, this would be WebRTC stream
      videoRef.current.srcObject = null;
    }
  }, [isVideoOff]);

  const getQualityColor = () => {
    switch (connectionQuality) {
      case 'excellent': return 'text-green-400';
      case 'good': return 'text-green-500';
      case 'fair': return 'text-yellow-500';
      case 'poor': return 'text-red-500';
      default: return 'text-gray-400';
    }
  };

  const getQualityBars = () => {
    const bars = [];
    const levels = { excellent: 4, good: 3, fair: 2, poor: 1 };
    const level = levels?.[connectionQuality] || 0;
    
    for (let i = 0; i < 4; i++) {
      bars?.push(
        <div
          key={i}
          className={`w-1 bg-current transition-all duration-300 ${
            i < level ? 'opacity-100' : 'opacity-30'
          } ${i === 0 ? 'h-2' : i === 1 ? 'h-3' : i === 2 ? 'h-4' : 'h-5'}`}
        />
      );
    }
    return bars;
  };

  return (
    <div className={`relative bg-gray-900 rounded-lg overflow-hidden shadow-elevation-2 ${
      isMainVideo ? 'h-full' : 'h-48'
    }`}>
      {/* Video Stream */}
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
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              muted={isMainVideo}
              playsInline
            />
            {/* Placeholder for demo */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-purple-600/20 flex items-center justify-center">
              <div className="text-center">
                <div className="w-20 h-20 bg-white/10 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Icon name="Video" size={40} className="text-white" />
                </div>
                <p className="text-white font-medium text-lg">{participantName}</p>
              </div>
            </div>
          </>
        )}

        {/* Connection Quality Indicator */}
        <div className={`absolute top-3 right-3 flex items-center gap-1 ${getQualityColor()}`}>
          <div className="flex items-end gap-0.5 h-5">
            {getQualityBars()}
          </div>
        </div>

        {/* Audio Muted Indicator */}
        {isAudioMuted && (
          <div className="absolute bottom-3 left-3 bg-red-600 rounded-full p-2">
            <Icon name="MicOff" size={16} className="text-white" />
          </div>
        )}

        {/* Screen Share Indicator */}
        {isScreenSharing && (
          <div className="absolute top-3 left-3 bg-blue-600 rounded-full px-3 py-1 flex items-center gap-2">
            <Icon name="Monitor" size={14} className="text-white" />
            <span className="text-white text-xs font-medium">Compartilhando tela</span>
          </div>
        )}

        {/* Participant Name Overlay */}
        <div className="absolute bottom-3 left-3 bg-black/50 backdrop-blur-sm rounded px-3 py-1">
          <span className="text-white text-sm font-medium">{participantName}</span>
        </div>

        {/* Controls for main video */}
        {isMainVideo && (
          <div className="absolute bottom-4 right-4 flex gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="bg-black/50 backdrop-blur-sm hover:bg-black/70 text-white"
              onClick={() => setIsFullscreen(!isFullscreen)}
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