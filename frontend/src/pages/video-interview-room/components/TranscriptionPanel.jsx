import React, { useState, useRef, useEffect } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const TranscriptionPanel = ({ isOpen, onToggle }) => {
  const [transcriptions, setTranscriptions] = useState([
    {
      id: 1,
      speaker: "Ana Silva",
      text: "Boa tarde, Carlos. Obrigada por estar aqui hoje. Vamos começar com uma pergunta sobre sua experiência.",
      timestamp: new Date(Date.now() - 300000),
      confidence: 0.95,
      type: "interviewer"
    },
    {
      id: 2,
      speaker: "Carlos Santos",
      text: "Boa tarde, Ana. Muito obrigado pela oportunidade. Estou animado para conversar sobre minha experiência em desenvolvimento.",
      timestamp: new Date(Date.now() - 280000),
      confidence: 0.92,
      type: "candidate"
    },
    {
      id: 3,
      speaker: "Ana Silva",
      text: "Perfeito. Conte-me sobre seus últimos projetos com React e como você estrutura componentes complexos.",
      timestamp: new Date(Date.now() - 240000),
      confidence: 0.88,
      type: "interviewer"
    },
    {
      id: 4,
      speaker: "Carlos Santos",
      text: "Nos meus projetos recentes, tenho focado em arquitetura de componentes reutilizáveis. Por exemplo, no último projeto...",
      timestamp: new Date(Date.now() - 200000),
      confidence: 0.91,
      type: "candidate"
    }
  ]);

  const [isRecording, setIsRecording] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const transcriptionEndRef = useRef(null);

  const scrollToBottom = () => {
    transcriptionEndRef?.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [transcriptions]);

  // Simulate real-time transcription
  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        const newTranscription = {
          id: Date.now(),
          speaker: Math.random() > 0.5 ? "Ana Silva" : "Carlos Santos",
          text: "Esta é uma transcrição simulada em tempo real...",
          timestamp: new Date(),
          confidence: 0.85 + Math.random() * 0.15,
          type: Math.random() > 0.5 ? "interviewer" : "candidate"
        };
        
        setTranscriptions(prev => [...prev, newTranscription]);
      }, 30000);

      return () => clearInterval(interval);
    }
  }, [isRecording]);

  const formatTime = (date) => {
    return date?.toLocaleTimeString('pt-BR', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-500';
    if (confidence >= 0.8) return 'text-yellow-500';
    return 'text-red-500';
  };

  const filteredTranscriptions = transcriptions?.filter(t =>
    t?.text?.toLowerCase()?.includes(searchTerm?.toLowerCase()) ||
    t?.speaker?.toLowerCase()?.includes(searchTerm?.toLowerCase())
  );

  const exportTranscription = () => {
    const content = transcriptions?.map(t => 
      `[${formatTime(t?.timestamp)}] ${t?.speaker}: ${t?.text}`
    )?.join('\n\n');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcricao-entrevista-${new Date()?.toISOString()?.split('T')?.[0]}.txt`;
    a?.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`fixed bottom-0 left-0 right-0 bg-card border-t border-border transition-all duration-300 ${
      isOpen ? 'h-80' : 'h-12'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="h-8 w-8"
          >
            <Icon name={isOpen ? "ChevronDown" : "ChevronUp"} size={16} />
          </Button>
          
          <div className="flex items-center gap-2">
            <Icon name="FileText" size={16} className="text-blue-500" />
            <span className="font-medium text-foreground">Transcrição em Tempo Real</span>
          </div>

          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-gray-400'}`}></div>
            <span className="text-sm text-muted-foreground">
              {isRecording ? 'Gravando' : 'Pausado'}
            </span>
          </div>
        </div>

        {isOpen && (
          <div className="flex items-center gap-2">
            <div className="relative">
              <Icon name="Search" size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                placeholder="Buscar na transcrição..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e?.target?.value)}
                className="pl-10 pr-4 py-2 bg-muted border border-border rounded-lg text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsRecording(!isRecording)}
            >
              <Icon name={isRecording ? "Pause" : "Play"} size={14} className="mr-2" />
              {isRecording ? 'Pausar' : 'Retomar'}
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={exportTranscription}
            >
              <Icon name="Download" size={14} className="mr-2" />
              Exportar
            </Button>
          </div>
        )}
      </div>
      {/* Transcription Content */}
      {isOpen && (
        <div className="flex-1 overflow-y-auto p-4 h-64">
          <div className="space-y-4">
            {filteredTranscriptions?.map((transcription) => (
              <div key={transcription?.id} className="flex gap-3">
                <div className="flex-shrink-0 w-16 text-xs text-muted-foreground text-right">
                  {formatTime(transcription?.timestamp)}
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`font-medium text-sm ${
                      transcription?.type === 'interviewer' ? 'text-blue-600' : 'text-green-600'
                    }`}>
                      {transcription?.speaker}
                    </span>
                    
                    <div className="flex items-center gap-1">
                      <Icon name="Volume2" size={12} className={getConfidenceColor(transcription?.confidence)} />
                      <span className={`text-xs ${getConfidenceColor(transcription?.confidence)}`}>
                        {Math.round(transcription?.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                  
                  <p className="text-foreground text-sm leading-relaxed">
                    {transcription?.text}
                  </p>
                </div>

                <div className="flex-shrink-0">
                  <Button variant="ghost" size="icon" className="h-6 w-6">
                    <Icon name="Copy" size={12} />
                  </Button>
                </div>
              </div>
            ))}
            
            <div ref={transcriptionEndRef} />
          </div>

          {filteredTranscriptions?.length === 0 && searchTerm && (
            <div className="text-center py-8">
              <Icon name="Search" size={48} className="text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">
                Nenhum resultado encontrado para "{searchTerm}"
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TranscriptionPanel;