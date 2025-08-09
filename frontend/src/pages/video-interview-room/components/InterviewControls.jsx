import React, { useState, useEffect } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const InterviewControls = ({
  isAudioMuted = false,
  isVideoOff = false,
  isScreenSharing = false,
  isRecording = false,
  onToggleAudio,
  onToggleVideo,
  onToggleScreenShare,
  onToggleRecording,
  onEndInterview
}) => {
  const [interviewDuration, setInterviewDuration] = useState(0);
  const [showEndConfirm, setShowEndConfirm] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setInterviewDuration(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours?.toString()?.padStart(2, '0')}:${minutes?.toString()?.padStart(2, '0')}:${secs?.toString()?.padStart(2, '0')}`;
    }
    return `${minutes?.toString()?.padStart(2, '0')}:${secs?.toString()?.padStart(2, '0')}`;
  };

  const handleEndInterview = () => {
    if (showEndConfirm) {
      onEndInterview();
    } else {
      setShowEndConfirm(true);
      setTimeout(() => setShowEndConfirm(false), 5000);
    }
  };

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
      <div className="bg-card/95 backdrop-blur-sm border border-border rounded-2xl px-6 py-4 shadow-elevation-3">
        <div className="flex items-center gap-4">
          {/* Interview Timer */}
          <div className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg">
            <Icon name="Clock" size={16} className="text-muted-foreground" />
            <span className="text-sm font-mono text-foreground">
              {formatDuration(interviewDuration)}
            </span>
          </div>

          {/* Audio Control */}
          <Button
            variant={isAudioMuted ? "destructive" : "outline"}
            size="icon"
            onClick={onToggleAudio}
            className="h-12 w-12 rounded-full"
          >
            <Icon name={isAudioMuted ? "MicOff" : "Mic"} size={20} />
          </Button>

          {/* Video Control */}
          <Button
            variant={isVideoOff ? "destructive" : "outline"}
            size="icon"
            onClick={onToggleVideo}
            className="h-12 w-12 rounded-full"
          >
            <Icon name={isVideoOff ? "VideoOff" : "Video"} size={20} />
          </Button>

          {/* Screen Share Control */}
          <Button
            variant={isScreenSharing ? "default" : "outline"}
            size="icon"
            onClick={onToggleScreenShare}
            className="h-12 w-12 rounded-full"
          >
            <Icon name="Monitor" size={20} />
          </Button>

          {/* Recording Control */}
          <Button
            variant={isRecording ? "destructive" : "outline"}
            size="icon"
            onClick={onToggleRecording}
            className="h-12 w-12 rounded-full relative"
          >
            <Icon name="Circle" size={20} />
            {isRecording && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
            )}
          </Button>

          {/* Divider */}
          <div className="w-px h-8 bg-border"></div>

          {/* Settings */}
          <Button
            variant="ghost"
            size="icon"
            className="h-12 w-12 rounded-full"
          >
            <Icon name="Settings" size={20} />
          </Button>

          {/* End Interview */}
          <Button
            variant={showEndConfirm ? "destructive" : "outline"}
            onClick={handleEndInterview}
            className="px-6 h-12 rounded-full"
          >
            <Icon name="PhoneOff" size={16} className="mr-2" />
            {showEndConfirm ? "Confirmar Encerramento" : "Encerrar"}
          </Button>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center justify-center gap-4 mt-3 pt-3 border-t border-border">
          {isRecording && (
            <div className="flex items-center gap-2 text-red-500">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-xs font-medium">Gravando</span>
            </div>
          )}
          
          {isScreenSharing && (
            <div className="flex items-center gap-2 text-blue-500">
              <Icon name="Monitor" size={12} />
              <span className="text-xs font-medium">Compartilhando tela</span>
            </div>
          )}

          <div className="flex items-center gap-2 text-green-500">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-xs font-medium">Conexão estável</span>
          </div>
        </div>

        {/* End Confirmation Message */}
        {showEndConfirm && (
          <div className="absolute -top-12 left-1/2 -translate-x-1/2 bg-destructive text-destructive-foreground px-4 py-2 rounded-lg text-sm whitespace-nowrap">
            Clique novamente para confirmar
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewControls;