import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import InterviewHeader from './components/InterviewHeader';
import VideoPanel from './components/VideoPanel';
import AIAvatarPanel from './components/AIAvatarPanel';
import LiveChat from './components/LiveChat';
import TranscriptionPanel from './components/TranscriptionPanel';
import InterviewControls from './components/InterviewControls';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';

const VideoInterviewRoom = () => {
  const navigate = useNavigate();
  
  // Interview state
  const [interviewState, setInterviewState] = useState({
    isAudioMuted: false,
    isVideoOff: false,
    isScreenSharing: false,
    isRecording: true,
    connectionQuality: 'good'
  });

  // UI state
  const [uiState, setUiState] = useState({
    isAIAvatarActive: false,
    isChatOpen: false,
    isTranscriptionOpen: true,
    showEmergencyOptions: false
  });

  // Interview data
  const interviewData = {
    candidateName: "Carlos Santos",
    position: "Desenvolvedor React Sênior",
    interviewType: "Técnica",
    startTime: new Date(Date.now() - 900000), // Started 15 minutes ago
    interviewerId: "Ana Silva",
    roomId: "interview-room-12345"
  };

  // Handle interview controls
  const handleToggleAudio = () => {
    setInterviewState(prev => ({
      ...prev,
      isAudioMuted: !prev?.isAudioMuted
    }));
  };

  const handleToggleVideo = () => {
    setInterviewState(prev => ({
      ...prev,
      isVideoOff: !prev?.isVideoOff
    }));
  };

  const handleToggleScreenShare = () => {
    setInterviewState(prev => ({
      ...prev,
      isScreenSharing: !prev?.isScreenSharing
    }));
  };

  const handleToggleRecording = () => {
    setInterviewState(prev => ({
      ...prev,
      isRecording: !prev?.isRecording
    }));
  };

  const handleEndInterview = () => {
    // In real implementation, this would clean up WebRTC connections
    navigate('/interview-transcription-analysis', {
      state: {
        candidateName: interviewData?.candidateName,
        position: interviewData?.position,
        duration: Math.floor((Date.now() - interviewData?.startTime?.getTime()) / 1000)
      }
    });
  };

  // Handle AI Avatar
  const handleToggleAIAvatar = () => {
    setUiState(prev => ({
      ...prev,
      isAIAvatarActive: !prev?.isAIAvatarActive
    }));
  };

  // Handle Chat
  const handleToggleChat = () => {
    setUiState(prev => ({
      ...prev,
      isChatOpen: !prev?.isChatOpen
    }));
  };

  // Handle Transcription
  const handleToggleTranscription = () => {
    setUiState(prev => ({
      ...prev,
      isTranscriptionOpen: !prev?.isTranscriptionOpen
    }));
  };

  // Monitor connection quality
  useEffect(() => {
    const interval = setInterval(() => {
      const qualities = ['excellent', 'good', 'fair', 'poor'];
      const randomQuality = qualities?.[Math.floor(Math.random() * qualities?.length)];
      setInterviewState(prev => ({
        ...prev,
        connectionQuality: randomQuality
      }));
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  // Emergency backup options
  const emergencyOptions = [
    {
      icon: "Phone",
      title: "Ligar por Telefone",
      description: "Conectar via chamada telefônica",
      action: () => console.log("Phone dial initiated")
    },
    {
      icon: "MessageSquare",
      title: "Apenas Chat",
      description: "Continuar apenas com mensagens",
      action: () => setUiState(prev => ({ ...prev, isChatOpen: true }))
    },
    {
      icon: "RefreshCw",
      title: "Reconectar",
      description: "Tentar reconectar vídeo",
      action: () => window.location?.reload()
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Interview Header */}
      <InterviewHeader
        candidateName={interviewData?.candidateName}
        position={interviewData?.position}
        interviewType={interviewData?.interviewType}
        startTime={interviewData?.startTime}
      />
      {/* Main Interview Area */}
      <div className="flex h-[calc(100vh-80px)]">
        {/* Left Sidebar - AI Avatar (when active) */}
        {uiState?.isAIAvatarActive && (
          <div className="w-80 border-r border-border p-4">
            <AIAvatarPanel
              isActive={uiState?.isAIAvatarActive}
              onToggle={handleToggleAIAvatar}
            />
          </div>
        )}

        {/* Main Video Area */}
        <div className="flex-1 p-4">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 h-full">
            {/* Candidate Video - Main */}
            <div className="lg:col-span-3 h-full">
              <VideoPanel
                isMainVideo={true}
                participantName={interviewData?.candidateName}
                isAudioMuted={false}
                isVideoOff={false}
                connectionQuality={interviewState?.connectionQuality}
                onToggleAudio={() => {}}
                onToggleVideo={() => {}}
                onToggleScreenShare={() => {}}
              />
            </div>

            {/* Right Panel */}
            <div className="space-y-4">
              {/* Interviewer Video */}
              <VideoPanel
                isMainVideo={false}
                participantName={interviewData?.interviewerId}
                isAudioMuted={interviewState?.isAudioMuted}
                isVideoOff={interviewState?.isVideoOff}
                connectionQuality="excellent"
                onToggleAudio={handleToggleAudio}
                onToggleVideo={handleToggleVideo}
                onToggleScreenShare={handleToggleScreenShare}
              />

              {/* AI Avatar Toggle (when not active) */}
              {!uiState?.isAIAvatarActive && (
                <div className="bg-card rounded-lg p-4 border border-border">
                  <div className="text-center">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-3">
                      <Icon name="Bot" size={24} className="text-white" />
                    </div>
                    <h3 className="font-semibold text-foreground mb-2">Assistente IA</h3>
                    <p className="text-sm text-muted-foreground mb-3">
                      Ative para receber sugestões
                    </p>
                    <Button
                      size="sm"
                      onClick={handleToggleAIAvatar}
                      className="w-full"
                    >
                      Ativar
                    </Button>
                  </div>
                </div>
              )}

              {/* Quick Stats */}
              <div className="bg-card rounded-lg p-4 border border-border">
                <h3 className="font-semibold text-foreground mb-3">Status da Entrevista</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Duração:</span>
                    <span className="text-foreground font-medium">15:30</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Qualidade:</span>
                    <span className={`font-medium capitalize ${
                      interviewState?.connectionQuality === 'excellent' ? 'text-green-500' :
                      interviewState?.connectionQuality === 'good' ? 'text-green-600' :
                      interviewState?.connectionQuality === 'fair'? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {interviewState?.connectionQuality === 'excellent' ? 'Excelente' :
                       interviewState?.connectionQuality === 'good' ? 'Boa' :
                       interviewState?.connectionQuality === 'fair' ? 'Regular' : 'Ruim'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Gravação:</span>
                    <span className={`font-medium ${interviewState?.isRecording ? 'text-red-500' : 'text-gray-500'}`}>
                      {interviewState?.isRecording ? 'Ativa' : 'Inativa'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Emergency Options */}
              <div className="bg-card rounded-lg p-4 border border-border">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={() => setUiState(prev => ({ ...prev, showEmergencyOptions: !prev?.showEmergencyOptions }))}
                >
                  <Icon name="AlertTriangle" size={16} className="mr-2" />
                  Opções de Emergência
                </Button>
                
                {uiState?.showEmergencyOptions && (
                  <div className="mt-3 space-y-2">
                    {emergencyOptions?.map((option, index) => (
                      <Button
                        key={index}
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start"
                        onClick={option?.action}
                      >
                        <Icon name={option?.icon} size={14} className="mr-2" />
                        <div className="text-left">
                          <div className="text-xs font-medium">{option?.title}</div>
                          <div className="text-xs text-muted-foreground">{option?.description}</div>
                        </div>
                      </Button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Live Chat */}
      <LiveChat
        isOpen={uiState?.isChatOpen}
        onToggle={handleToggleChat}
      />
      {/* Transcription Panel */}
      <TranscriptionPanel
        isOpen={uiState?.isTranscriptionOpen}
        onToggle={handleToggleTranscription}
      />
      {/* Interview Controls */}
      <InterviewControls
        isAudioMuted={interviewState?.isAudioMuted}
        isVideoOff={interviewState?.isVideoOff}
        isScreenSharing={interviewState?.isScreenSharing}
        isRecording={interviewState?.isRecording}
        onToggleAudio={handleToggleAudio}
        onToggleVideo={handleToggleVideo}
        onToggleScreenShare={handleToggleScreenShare}
        onToggleRecording={handleToggleRecording}
        onEndInterview={handleEndInterview}
      />
      {/* Recording Privacy Notice */}
      {interviewState?.isRecording && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 bg-red-600 text-white px-4 py-2 rounded-lg shadow-lg z-40">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            <span className="text-sm font-medium">Esta entrevista está sendo gravada</span>
          </div>
        </div>
      )}
      {/* Connection Issues Warning */}
      {interviewState?.connectionQuality === 'poor' && (
        <div className="fixed top-16 left-1/2 -translate-x-1/2 bg-yellow-600 text-white px-4 py-2 rounded-lg shadow-lg z-40">
          <div className="flex items-center gap-2">
            <Icon name="Wifi" size={16} />
            <span className="text-sm font-medium">Conexão instável detectada</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoInterviewRoom;