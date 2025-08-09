import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';
import TranscriptionPanel from './components/TranscriptionPanel';
import EmotionalAnalysisChart from './components/EmotionalAnalysisChart';
import TechnicalSkillsRadar from './components/TechnicalSkillsRadar';
import SoftSkillsAssessment from './components/SoftSkillsAssessment';
import KeyTopicsExtraction from './components/KeyTopicsExtraction';
import InterviewerNotes from './components/InterviewerNotes';
import CandidateProgressActions from './components/CandidateProgressActions';

const InterviewTranscriptionAnalysis = () => {
  const [activeTab, setActiveTab] = useState('transcription');
  const [selectedTimestamp, setSelectedTimestamp] = useState(null);
  const [interviewerNotes, setInterviewerNotes] = useState([]);

  // Mock data for interview transcription
  const mockTranscription = [
    {
      timestamp: 15,
      speaker: "Entrevistador",
      text: "Bom dia, Maria! Obrigado por estar aqui hoje. Pode nos contar um pouco sobre sua experiência com desenvolvimento React?",
      confidence: 95,
      highlighted: false
    },
    {
      timestamp: 32,
      speaker: "Candidato",
      text: "Bom dia! Tenho 4 anos de experiência com React, trabalhei em projetos desde aplicações simples até sistemas complexos com Redux e Context API. Recentemente implementei uma arquitetura de micro-frontends usando React 18.",
      confidence: 92,
      highlighted: true
    },
    {
      timestamp: 78,
      speaker: "Entrevistador",
      text: "Interessante! Como você lida com otimização de performance em aplicações React grandes?",
      confidence: 96,
      highlighted: false
    },
    {
      timestamp: 95,
      speaker: "Candidato",
      text: "Uso várias estratégias: lazy loading com React.lazy, memoização com useMemo e useCallback, virtualização para listas grandes, e code splitting. Também monitoro com React DevTools Profiler para identificar gargalos.",
      confidence: 89,
      highlighted: true
    },
    {
      timestamp: 145,
      speaker: "Entrevistador",
      text: "Excelente! Agora, me conte sobre um desafio técnico complexo que você enfrentou recentemente.",
      confidence: 94,
      highlighted: false
    },
    {
      timestamp: 162,
      speaker: "Candidato",
      text: "Tivemos um problema de memory leak em uma aplicação com muitos componentes dinâmicos. Identifiquei que event listeners não estavam sendo removidos corretamente. Implementei um hook customizado para gerenciar cleanup automático e reduzi o uso de memória em 40%.",
      confidence: 91,
      highlighted: true
    }
  ];

  // Mock data for emotional analysis
  const mockEmotionalData = [
    { emotion: "Confiança", intensity: 85 },
    { emotion: "Entusiasmo", intensity: 78 },
    { emotion: "Nervosismo", intensity: 25 },
    { emotion: "Clareza", intensity: 92 },
    { emotion: "Assertividade", intensity: 80 },
    { emotion: "Empatia", intensity: 70 }
  ];

  // Mock data for technical skills
  const mockTechnicalSkills = [
    { skill: "React/JSX", proficiency: 90 },
    { skill: "JavaScript ES6+", proficiency: 88 },
    { skill: "TypeScript", proficiency: 75 },
    { skill: "Node.js", proficiency: 70 },
    { skill: "Redux/Context", proficiency: 85 },
    { skill: "Testing", proficiency: 65 },
    { skill: "Git/GitHub", proficiency: 80 },
    { skill: "CSS/Styling", proficiency: 78 }
  ];

  // Mock data for soft skills
  const mockSoftSkills = [
    {
      name: "Liderança",
      score: 75,
      description: "Capacidade de guiar e motivar equipes",
      feedback: "Demonstrou experiência em liderar projetos técnicos"
    },
    {
      name: "Trabalho em Equipe",
      score: 88,
      description: "Colaboração efetiva com colegas",
      feedback: "Excelente comunicação e disposição para ajudar outros"
    },
    {
      name: "Resolução de Problemas",
      score: 92,
      description: "Análise e solução de desafios complexos",
      feedback: "Abordagem sistemática e criativa para resolver problemas"
    },
    {
      name: "Adaptabilidade",
      score: 80,
      description: "Flexibilidade para mudanças e novos desafios",
      feedback: "Boa capacidade de se adaptar a novas tecnologias"
    },
    {
      name: "Comunicação",
      score: 85,
      description: "Clareza na expressão de ideias",
      feedback: "Comunicação técnica clara e objetiva"
    },
    {
      name: "Criatividade",
      score: 70,
      description: "Pensamento inovador e soluções originais",
      feedback: "Apresentou algumas soluções criativas interessantes"
    }
  ];

  // Mock data for key topics
  const mockTopics = [
    {
      title: "Experiência com React",
      description: "Discussão sobre conhecimentos e projetos em React",
      relevance: 95,
      keywords: ["React", "JSX", "Hooks", "Components", "State Management"],
      duration: 8,
      mentions: 12
    },
    {
      title: "Otimização de Performance",
      description: "Estratégias para melhorar performance de aplicações",
      relevance: 88,
      keywords: ["Performance", "Lazy Loading", "Memoization", "Code Splitting"],
      duration: 5,
      mentions: 8
    },
    {
      title: "Resolução de Problemas",
      description: "Exemplos de desafios técnicos enfrentados",
      relevance: 85,
      keywords: ["Memory Leak", "Debugging", "Problem Solving", "Optimization"],
      duration: 6,
      mentions: 7
    },
    {
      title: "Trabalho em Equipe",
      description: "Experiências de colaboração e liderança",
      relevance: 70,
      keywords: ["Team Work", "Collaboration", "Leadership", "Communication"],
      duration: 4,
      mentions: 5
    }
  ];

  // Mock candidate data
  const mockCandidate = {
    id: 1,
    name: "Maria Silva Santos",
    position: "Desenvolvedora React Sênior",
    status: "interviewing",
    overallScore: 84,
    interviewDate: "2025-01-08",
    interviewer: "João Mendes"
  };

  const tabs = [
    { id: 'transcription', label: 'Transcrição', icon: 'FileText' },
    { id: 'analysis', label: 'Análise IA', icon: 'Brain' },
    { id: 'skills', label: 'Habilidades', icon: 'Target' },
    { id: 'notes', label: 'Anotações', icon: 'StickyNote' },
    { id: 'actions', label: 'Ações', icon: 'Settings' }
  ];

  const handleTimestampClick = (timestamp) => {
    setSelectedTimestamp(timestamp);
    // Here you would typically seek to the timestamp in a video/audio player
    console.log(`Seeking to timestamp: ${timestamp}s`);
  };

  const handleAddNote = (note) => {
    setInterviewerNotes(prev => [...prev, note]);
  };

  const handleUpdateNote = (noteId, newText) => {
    setInterviewerNotes(prev => 
      prev?.map(note => 
        note?.id === noteId ? { ...note, text: newText } : note
      )
    );
  };

  const handleDeleteNote = (noteId) => {
    setInterviewerNotes(prev => prev?.filter(note => note?.id !== noteId));
  };

  const handleStatusChange = (candidateId, newStatus) => {
    console.log(`Changing candidate ${candidateId} status to: ${newStatus}`);
    // Here you would update the candidate status in your backend
  };

  const handleGenerateReport = (reportType) => {
    console.log(`Generating ${reportType} report`);
    // Here you would generate and download the report
  };

  // Initialize with some mock notes
  useEffect(() => {
    setInterviewerNotes([
      {
        id: 1,
        text: "Candidata demonstrou excelente conhecimento técnico em React. Respostas muito detalhadas e práticas.",
        author: "João Mendes",
        timestamp: new Date(Date.now() - 3600000),
        private: false
      },
      {
        id: 2,
        text: "Boa capacidade de comunicação técnica. Consegue explicar conceitos complexos de forma clara.",
        author: "Ana Costa",
        timestamp: new Date(Date.now() - 1800000),
        private: true
      }
    ]);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-card border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                to="/candidate-management"
                className="p-2 hover:bg-muted rounded-lg transition-colors"
              >
                <Icon name="ArrowLeft" size={20} className="text-muted-foreground" />
              </Link>
              <div>
                <h1 className="text-xl font-semibold text-foreground">
                  Análise de Entrevista
                </h1>
                <p className="text-sm text-muted-foreground">
                  {mockCandidate?.name} • {mockCandidate?.position}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="text-sm text-muted-foreground">Pontuação Geral</div>
                <div className="text-lg font-semibold text-primary">
                  {mockCandidate?.overallScore}/100
                </div>
              </div>
              <Button variant="outline" iconName="Share2">
                Compartilhar
              </Button>
            </div>
          </div>
        </div>
      </div>
      {/* Navigation */}
      <div className="bg-card border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {tabs?.map((tab) => (
              <button
                key={tab?.id}
                onClick={() => setActiveTab(tab?.id)}
                className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab?.id
                    ? 'border-primary text-primary' :'border-transparent text-muted-foreground hover:text-foreground hover:border-muted'
                }`}
              >
                <Icon name={tab?.icon} size={16} />
                {tab?.label}
              </button>
            ))}
          </nav>
        </div>
      </div>
      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'transcription' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
              <TranscriptionPanel 
                transcription={mockTranscription}
                onTimestampClick={handleTimestampClick}
              />
            </div>
            <div className="space-y-6">
              <EmotionalAnalysisChart emotionalData={mockEmotionalData} />
              <KeyTopicsExtraction topicsData={mockTopics} />
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <EmotionalAnalysisChart emotionalData={mockEmotionalData} />
            <KeyTopicsExtraction topicsData={mockTopics} />
          </div>
        )}

        {activeTab === 'skills' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <TechnicalSkillsRadar skillsData={mockTechnicalSkills} />
            <SoftSkillsAssessment softSkillsData={mockSoftSkills} />
          </div>
        )}

        {activeTab === 'notes' && (
          <div className="max-w-4xl mx-auto">
            <InterviewerNotes
              notes={interviewerNotes}
              onAddNote={handleAddNote}
              onUpdateNote={handleUpdateNote}
              onDeleteNote={handleDeleteNote}
            />
          </div>
        )}

        {activeTab === 'actions' && (
          <div className="max-w-2xl mx-auto">
            <CandidateProgressActions
              candidate={mockCandidate}
              onStatusChange={handleStatusChange}
              onGenerateReport={handleGenerateReport}
            />
          </div>
        )}
      </div>
      {/* Quick Navigation Sidebar */}
      <div className="hidden xl:block fixed right-8 top-1/2 transform -translate-y-1/2 z-40">
        <div className="bg-card border border-border rounded-lg p-2 shadow-lg">
          <div className="space-y-2">
            {[
              { route: '/job-posting-creation', icon: 'Plus', label: 'Criar Vaga' },
              { route: '/candidate-management', icon: 'Users', label: 'Candidatos' },
              { route: '/ai-resume-analysis', icon: 'FileSearch', label: 'Análise CV' },
              { route: '/video-interview-room', icon: 'Video', label: 'Entrevista' },
              { route: '/recruitment-analytics-dashboard', icon: 'BarChart3', label: 'Analytics' }
            ]?.map((item) => (
              <Link
                key={item?.route}
                to={item?.route}
                className="flex items-center justify-center w-10 h-10 rounded-lg hover:bg-muted transition-colors group relative"
                title={item?.label}
              >
                <Icon name={item?.icon} size={18} className="text-muted-foreground group-hover:text-foreground" />
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default InterviewTranscriptionAnalysis;