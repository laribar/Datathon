import React, { useState, useEffect } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const AIAvatarPanel = ({ isActive = false, onToggle }) => {
  const [currentSuggestion, setCurrentSuggestion] = useState(0);
  const [isListening, setIsListening] = useState(false);

  const suggestions = [
    "Conte-me sobre sua experiência com desenvolvimento React.",
    "Como você lidaria com um projeto com prazo apertado?",
    "Qual foi seu maior desafio técnico recentemente?",
    "Como você se mantém atualizado com novas tecnologias?",
    "Descreva uma situação onde teve que trabalhar em equipe."
  ];

  const avatarResponses = [
    "Baseado na análise do currículo, o candidato tem 3 anos de experiência em React.",
    "Recomendo explorar mais sobre metodologias ágeis nesta pergunta.",
    "O candidato demonstra conhecimento sólido em JavaScript moderno.",
    "Sugiro perguntar sobre projetos específicos mencionados no portfólio.",
    "Esta é uma boa oportunidade para avaliar soft skills."
  ];

  useEffect(() => {
    if (isActive) {
      const interval = setInterval(() => {
        setCurrentSuggestion(prev => (prev + 1) % suggestions?.length);
      }, 15000);
      return () => clearInterval(interval);
    }
  }, [isActive, suggestions?.length]);

  if (!isActive) {
    return (
      <div className="bg-card rounded-lg p-6 border border-border">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <Icon name="Bot" size={32} className="text-white" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">Assistente IA</h3>
          <p className="text-muted-foreground text-sm mb-4">
            Ative o assistente para receber sugestões de perguntas e análises em tempo real.
          </p>
          <Button onClick={onToggle} className="w-full">
            <Icon name="Play" size={16} className="mr-2" />
            Ativar Assistente
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      {/* Avatar Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
              <Icon name="Bot" size={20} className="text-white" />
            </div>
            <div>
              <h3 className="text-white font-semibold">Assistente IA</h3>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-white/80 text-xs">Ativo</span>
              </div>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="text-white hover:bg-white/20"
            onClick={onToggle}
          >
            <Icon name="X" size={16} />
          </Button>
        </div>
      </div>
      {/* Avatar Content */}
      <div className="p-4 space-y-4">
        {/* Current Analysis */}
        <div className="bg-muted/50 rounded-lg p-3">
          <div className="flex items-start gap-2 mb-2">
            <Icon name="Brain" size={16} className="text-blue-500 mt-0.5" />
            <span className="text-sm font-medium text-foreground">Análise Atual</span>
          </div>
          <p className="text-sm text-muted-foreground">
            {avatarResponses?.[currentSuggestion]}
          </p>
        </div>

        {/* Suggested Question */}
        <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
          <div className="flex items-start gap-2 mb-2">
            <Icon name="MessageSquare" size={16} className="text-blue-600 mt-0.5" />
            <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
              Pergunta Sugerida
            </span>
          </div>
          <p className="text-sm text-blue-700 dark:text-blue-300 mb-3">
            {suggestions?.[currentSuggestion]}
          </p>
          <Button size="sm" variant="outline" className="w-full">
            <Icon name="Copy" size={14} className="mr-2" />
            Copiar Pergunta
          </Button>
        </div>

        {/* Voice Recognition */}
        <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
          <div className="flex items-center gap-2">
            <Icon 
              name={isListening ? "Mic" : "MicOff"} 
              size={16} 
              className={isListening ? "text-green-500" : "text-gray-400"} 
            />
            <span className="text-sm text-foreground">
              {isListening ? "Ouvindo..." : "Reconhecimento de voz"}
            </span>
          </div>
          <Button
            size="sm"
            variant={isListening ? "destructive" : "default"}
            onClick={() => setIsListening(!isListening)}
          >
            {isListening ? "Parar" : "Iniciar"}
          </Button>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-2 gap-2">
          <Button size="sm" variant="outline">
            <Icon name="FileText" size={14} className="mr-2" />
            Notas
          </Button>
          <Button size="sm" variant="outline">
            <Icon name="Star" size={14} className="mr-2" />
            Avaliar
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AIAvatarPanel;