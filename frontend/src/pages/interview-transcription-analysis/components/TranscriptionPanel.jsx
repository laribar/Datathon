import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const TranscriptionPanel = ({ transcription, onTimestampClick }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSpeaker, setSelectedSpeaker] = useState('all');

  const speakers = ['all', 'Entrevistador', 'Candidato'];

  const filteredTranscription = transcription?.filter(item => {
    const matchesSearch = item?.text?.toLowerCase()?.includes(searchTerm?.toLowerCase());
    const matchesSpeaker = selectedSpeaker === 'all' || item?.speaker === selectedSpeaker;
    return matchesSearch && matchesSpeaker;
  });

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins?.toString()?.padStart(2, '0')}:${secs?.toString()?.padStart(2, '0')}`;
  };

  return (
    <div className="bg-card rounded-lg border border-border h-full flex flex-col">
      <div className="p-6 border-b border-border">
        <h2 className="text-xl font-semibold text-foreground mb-4">Transcrição da Entrevista</h2>
        
        {/* Search and Filter Controls */}
        <div className="space-y-4">
          <div className="relative">
            <Icon name="Search" size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Buscar na transcrição..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e?.target?.value)}
              className="w-full pl-10 pr-4 py-2 bg-input border border-border rounded-lg text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
          
          <div className="flex gap-2">
            {speakers?.map(speaker => (
              <Button
                key={speaker}
                variant={selectedSpeaker === speaker ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedSpeaker(speaker)}
              >
                {speaker === 'all' ? 'Todos' : speaker}
              </Button>
            ))}
          </div>
        </div>
      </div>
      {/* Transcription Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {filteredTranscription?.map((item, index) => (
          <div key={index} className="group">
            <div className="flex items-start gap-3">
              <button
                onClick={() => onTimestampClick(item?.timestamp)}
                className="text-primary hover:text-primary/80 text-sm font-mono min-w-[60px] mt-1 transition-colors"
              >
                {formatTime(item?.timestamp)}
              </button>
              
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-sm font-medium ${
                    item?.speaker === 'Entrevistador' ? 'text-accent' : 'text-success'
                  }`}>
                    {item?.speaker}
                  </span>
                  {item?.highlighted && (
                    <span className="px-2 py-1 bg-warning/20 text-warning text-xs rounded-full">
                      Destaque
                    </span>
                  )}
                </div>
                
                <p className={`text-foreground leading-relaxed ${
                  searchTerm && item?.text?.toLowerCase()?.includes(searchTerm?.toLowerCase())
                    ? 'bg-warning/20 rounded px-1' :''
                }`}>
                  {item?.text}
                </p>
                
                {item?.confidence && (
                  <div className="mt-2 flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Confiança:</span>
                    <div className="w-16 h-1 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-success rounded-full transition-all duration-300"
                        style={{ width: `${item?.confidence}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground">{item?.confidence}%</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        
        {filteredTranscription?.length === 0 && (
          <div className="text-center py-12">
            <Icon name="FileText" size={48} className="mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">
              {searchTerm ? 'Nenhum resultado encontrado' : 'Nenhuma transcrição disponível'}
            </p>
          </div>
        )}
      </div>
      {/* Export Controls */}
      <div className="p-6 border-t border-border">
        <div className="flex gap-2">
          <Button variant="outline" size="sm" iconName="Download">
            Exportar TXT
          </Button>
          <Button variant="outline" size="sm" iconName="FileText">
            Exportar PDF
          </Button>
        </div>
      </div>
    </div>
  );
};

export default TranscriptionPanel;