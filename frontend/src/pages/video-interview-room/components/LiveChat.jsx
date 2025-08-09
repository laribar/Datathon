import React, { useState, useRef, useEffect } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';
import Input from '../../../components/ui/Input';

const LiveChat = ({ isOpen, onToggle }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: "Sistema",
      message: "Entrevista iniciada às 14:30",
      timestamp: new Date(Date.now() - 300000),
      type: "system"
    },
    {
      id: 2,
      sender: "Ana Silva",
      message: "Boa tarde! Estou pronta para começar.",
      timestamp: new Date(Date.now() - 240000),
      type: "interviewer"
    },
    {
      id: 3,
      sender: "Carlos Santos",
      message: "Boa tarde! Obrigado pela oportunidade.",
      timestamp: new Date(Date.now() - 180000),
      type: "candidate"
    }
  ]);
  
  const [newMessage, setNewMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef?.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (e) => {
    e?.preventDefault();
    if (!newMessage?.trim()) return;

    const message = {
      id: Date.now(),
      sender: "Você",
      message: newMessage?.trim(),
      timestamp: new Date(),
      type: "interviewer"
    };

    setMessages(prev => [...prev, message]);
    setNewMessage("");

    // Simulate typing indicator
    setIsTyping(true);
    setTimeout(() => {
      setIsTyping(false);
    }, 2000);
  };

  const formatTime = (date) => {
    return date?.toLocaleTimeString('pt-BR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getMessageStyle = (type) => {
    switch (type) {
      case 'system':
        return 'bg-muted/50 text-muted-foreground text-center text-xs py-2';
      case 'interviewer':
        return 'bg-blue-600 text-white ml-auto max-w-[80%]';
      case 'candidate':
        return 'bg-muted text-foreground mr-auto max-w-[80%]';
      default:
        return 'bg-muted text-foreground mr-auto max-w-[80%]';
    }
  };

  if (!isOpen) {
    return (
      <Button
        onClick={onToggle}
        className="fixed right-4 top-1/2 -translate-y-1/2 z-50"
        size="icon"
      >
        <Icon name="MessageSquare" size={20} />
      </Button>
    );
  }

  return (
    <div className="fixed right-4 top-4 bottom-4 w-80 bg-card border border-border rounded-lg shadow-elevation-3 flex flex-col z-50">
      {/* Chat Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <Icon name="MessageSquare" size={20} className="text-blue-500" />
          <h3 className="font-semibold text-foreground">Chat da Entrevista</h3>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-xs text-muted-foreground">3 online</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="h-8 w-8"
          >
            <Icon name="X" size={16} />
          </Button>
        </div>
      </div>
      {/* Messages Container */}
      <div 
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-3"
      >
        {messages?.map((msg) => (
          <div key={msg?.id} className="flex flex-col">
            {msg?.type !== 'system' && (
              <div className={`flex items-center gap-2 mb-1 ${
                msg?.type === 'interviewer' ? 'justify-end' : 'justify-start'
              }`}>
                <span className="text-xs text-muted-foreground">
                  {msg?.sender}
                </span>
                <span className="text-xs text-muted-foreground">
                  {formatTime(msg?.timestamp)}
                </span>
              </div>
            )}
            <div className={`rounded-lg px-3 py-2 ${getMessageStyle(msg?.type)}`}>
              {msg?.message}
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-current rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            </div>
            <span className="text-xs">Alguém está digitando...</span>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      {/* Message Input */}
      <form onSubmit={handleSendMessage} className="p-4 border-t border-border">
        <div className="flex gap-2">
          <Input
            type="text"
            placeholder="Digite sua mensagem..."
            value={newMessage}
            onChange={(e) => setNewMessage(e?.target?.value)}
            className="flex-1"
          />
          <Button type="submit" size="icon" disabled={!newMessage?.trim()}>
            <Icon name="Send" size={16} />
          </Button>
        </div>
        
        {/* Quick Actions */}
        <div className="flex gap-2 mt-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setNewMessage("Podemos fazer uma pausa de 5 minutos?")}
          >
            <Icon name="Clock" size={14} className="mr-1" />
            Pausa
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => setNewMessage("Obrigado pela participação!")}
          >
            <Icon name="ThumbsUp" size={14} className="mr-1" />
            Obrigado
          </Button>
        </div>
      </form>
    </div>
  );
};

export default LiveChat;