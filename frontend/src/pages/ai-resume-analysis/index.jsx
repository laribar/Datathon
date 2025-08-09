import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';
import JobRequirementsPanel from './components/JobRequirementsPanel';
import CandidateQualificationsPanel from './components/CandidateQualificationsPanel';
import CompatibilityScoreChart from './components/CompatibilityScoreChart';
import TechnicalSkillsRadar from './components/TechnicalSkillsRadar';
import ResumeViewer from './components/ResumeViewer';
import SkillsMatrix from './components/SkillsMatrix';
import RecommendationEngine from './components/RecommendationEngine';
import ActionButtons from './components/ActionButtons';

const AIResumeAnalysis = () => {
  const [currentLanguage, setCurrentLanguage] = useState('pt');

  useEffect(() => {
    const savedLanguage = localStorage.getItem('language') || 'pt';
    setCurrentLanguage(savedLanguage);
  }, []);

  // Mock data for job requirements
  const jobData = {
    title: "Desenvolvedor React Sênior",
    description: `Estamos procurando um desenvolvedor React experiente para liderar o desenvolvimento de aplicações web modernas. O candidato ideal deve ter sólida experiência em React, TypeScript, e arquitetura de frontend, além de habilidades em liderança técnica e mentoria de equipe.`,
    requiredSkills: ["React", "TypeScript", "JavaScript", "Node.js", "GraphQL", "Jest", "CSS", "HTML", "Git", "Agile"],
    experienceLevel: "5+ anos em desenvolvimento frontend",
    education: "Superior em Ciência da Computação ou área relacionada",
    location: "São Paulo, SP (Híbrido)",
    salaryRange: "R$ 12.000 - R$ 18.000"
  };

  // Mock data for candidate
  const candidateData = {
    name: "Ana Silva Santos",
    position: "Desenvolvedora Frontend Sênior",
    avatar: "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face",
    email: "ana.silva@email.com",
    phone: "+55 11 99999-9999",
    location: "São Paulo, SP",
    status: "screening",
    summary: `Desenvolvedora frontend com 6 anos de experiência em React e TypeScript. Especialista em arquitetura de componentes, otimização de performance e liderança técnica. Experiência em projetos de grande escala e mentoria de equipes.`,
    skills: ["React", "TypeScript", "JavaScript", "Node.js", "GraphQL", "Jest", "CSS", "HTML", "Git", "Docker", "AWS"],
    experience: [
      {
        position: "Desenvolvedora Frontend Sênior",
        company: "TechCorp Brasil",
        duration: "2021 - Atual",
        description: "Liderança técnica em projetos React, mentoria de 3 desenvolvedores júnior"
      },
      {
        position: "Desenvolvedora Frontend Pleno",
        company: "StartupXYZ",
        duration: "2019 - 2021",
        description: "Desenvolvimento de aplicações React com TypeScript e GraphQL"
      },
      {
        position: "Desenvolvedora Frontend Júnior",
        company: "WebSolutions",
        duration: "2018 - 2019",
        description: "Desenvolvimento de interfaces responsivas com React e CSS"
      }
    ],
    education: [
      {
        degree: "Bacharelado em Ciência da Computação",
        institution: "Universidade de São Paulo",
        year: "2018"
      },
      {
        degree: "Certificação React Advanced",
        institution: "Meta",
        year: "2022"
      }
    ]
  };

  // Mock compatibility analysis
  const compatibilityScore = 87;
  const scoreBreakdown = {
    technical: 92,
    experience: 85,
    education: 84
  };

  // Keywords analysis
  const matchedKeywords = ["react", "typescript", "javascript", "node.js", "graphql", "jest", "css", "html", "git"];
  const missingKeywords = ["agile"];
  const highlightedKeywords = [...matchedKeywords];

  // Technical skills radar data
  const technicalSkillsData = [
    {
      name: "React",
      score: 95,
      level: "Expert",
      experience: "6 anos",
      description: "Desenvolvimento avançado com hooks, context, e otimização",
      required: true
    },
    {
      name: "TypeScript",
      score: 90,
      level: "Avançado",
      experience: "4 anos",
      description: "Tipagem avançada, generics, e arquitetura type-safe",
      required: true
    },
    {
      name: "JavaScript",
      score: 88,
      level: "Avançado",
      experience: "6 anos",
      description: "ES6+, async/await, e padrões modernos",
      required: true
    },
    {
      name: "Node.js",
      score: 75,
      level: "Intermediário",
      experience: "3 anos",
      description: "APIs REST, Express, e integração com bancos de dados",
      required: false
    },
    {
      name: "GraphQL",
      score: 80,
      level: "Intermediário",
      experience: "2 anos",
      description: "Queries, mutations, e integração com Apollo Client",
      required: false
    },
    {
      name: "Testing",
      score: 85,
      level: "Avançado",
      experience: "4 anos",
      description: "Jest, React Testing Library, e TDD",
      required: true
    }
  ];

  // Skills matrix data
  const requiredSkills = [
    { name: "React", level: 85 },
    { name: "TypeScript", level: 80 },
    { name: "JavaScript", level: 75 },
    { name: "Node.js", level: 60 },
    { name: "GraphQL", level: 50 },
    { name: "Jest", level: 70 }
  ];

  const candidateSkills = [
    { name: "React", level: 95, proficiency: "Expert" },
    { name: "TypeScript", level: 90, proficiency: "Avançado" },
    { name: "JavaScript", level: 88, proficiency: "Avançado" },
    { name: "Node.js", level: 75, proficiency: "Intermediário" },
    { name: "GraphQL", level: 80, proficiency: "Intermediário" },
    { name: "Jest", level: 85, proficiency: "Avançado" }
  ];

  // Recommendations data
  const recommendations = {
    overall: `Ana é uma candidata excepcional com forte compatibilidade técnica (87%). Suas habilidades em React e TypeScript excedem os requisitos, e sua experiência em liderança técnica é valiosa. Recomendo prosseguir com entrevista técnica focada em arquitetura e liderança.`
  };

  const candidateStrengths = [
    "Experiência sólida em React e TypeScript",
    "Liderança técnica comprovada",
    "Conhecimento em testes automatizados",
    "Experiência com projetos de grande escala"
  ];

  const candidateWeaknesses = [
    "Experiência limitada com metodologias ágeis",
    "Conhecimento intermediário em Node.js",
    "Pode precisar de mentoria em GraphQL avançado"
  ];

  // Action handlers
  const handleStatusChange = (newStatus) => {
    console.log(`Status changed to: ${newStatus}`);
    // Here you would typically update the candidate status via API
  };

  const handleScheduleInterview = () => {
    console.log('Scheduling interview...');
    // Here you would typically navigate to interview scheduling
  };

  const handleGenerateReport = () => {
    console.log('Generating report...');
    // Here you would typically generate and download a PDF report
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="bg-card border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-2">
                <Icon name="Brain" size={24} className="text-primary" />
                <span className="text-xl font-bold text-foreground">AI Recruitment</span>
              </div>
              
              <div className="hidden md:flex items-center gap-6">
                <Link
                  to="/job-posting-creation"
                  className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Icon name="Plus" size={16} />
                  Criar Vaga
                </Link>
                <Link
                  to="/candidate-management"
                  className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Icon name="Users" size={16} />
                  Candidatos
                </Link>
                <Link
                  to="/ai-resume-analysis"
                  className="flex items-center gap-2 px-3 py-2 text-sm text-primary font-medium border-b-2 border-primary"
                >
                  <Icon name="Brain" size={16} />
                  Análise IA
                </Link>
                <Link
                  to="/video-interview-room"
                  className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Icon name="Video" size={16} />
                  Entrevistas
                </Link>
                <Link
                  to="/interview-transcription-analysis"
                  className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Icon name="FileText" size={16} />
                  Transcrições
                </Link>
                <Link
                  to="/recruitment-analytics-dashboard"
                  className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Icon name="BarChart3" size={16} />
                  Analytics
                </Link>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <Button variant="ghost" size="sm" iconName="Bell" />
              <Button variant="ghost" size="sm" iconName="Settings" />
              <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                <span className="text-xs font-medium text-primary-foreground">HR</span>
              </div>
            </div>
          </div>
        </div>
      </nav>
      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <Button
              variant="ghost"
              size="sm"
              iconName="ArrowLeft"
              onClick={() => window.history?.back()}
            />
            <div>
              <h1 className="text-3xl font-bold text-foreground">Análise IA de Currículo</h1>
              <p className="text-muted-foreground">
                Análise detalhada usando algoritmos SBERT e LogReg para decisões baseadas em dados
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Icon name="Calendar" size={14} />
              <span>Analisado em {new Date()?.toLocaleDateString('pt-BR')}</span>
            </div>
            <div className="flex items-center gap-1">
              <Icon name="Clock" size={14} />
              <span>Tempo de análise: 2.3s</span>
            </div>
            <div className="flex items-center gap-1">
              <Icon name="Zap" size={14} />
              <span>IA Score: {compatibilityScore}%</span>
            </div>
          </div>
        </div>

        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          {/* Left Column - Job Requirements */}
          <div className="xl:col-span-4">
            <JobRequirementsPanel
              jobData={jobData}
              matchedKeywords={matchedKeywords}
              missingKeywords={missingKeywords}
            />
          </div>

          {/* Center Column - Compatibility Score */}
          <div className="xl:col-span-4">
            <CompatibilityScoreChart
              score={compatibilityScore}
              breakdown={scoreBreakdown}
            />
          </div>

          {/* Right Column - Candidate Qualifications */}
          <div className="xl:col-span-4">
            <CandidateQualificationsPanel
              candidateData={candidateData}
              matchedKeywords={matchedKeywords}
              missingKeywords={missingKeywords}
            />
          </div>
        </div>

        {/* Second Row - Technical Skills Radar */}
        <div className="mt-6">
          <TechnicalSkillsRadar skillsData={technicalSkillsData} />
        </div>

        {/* Third Row - Resume Viewer and Skills Matrix */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          <ResumeViewer
            resumeData={candidateData}
            highlightedKeywords={highlightedKeywords}
          />
          <SkillsMatrix
            requiredSkills={requiredSkills}
            candidateSkills={candidateSkills}
          />
        </div>

        {/* Fourth Row - Recommendations and Actions */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          <RecommendationEngine
            recommendations={recommendations}
            candidateStrengths={candidateStrengths}
            candidateWeaknesses={candidateWeaknesses}
          />
          <ActionButtons
            candidateData={candidateData}
            onStatusChange={handleStatusChange}
            onScheduleInterview={handleScheduleInterview}
            onGenerateReport={handleGenerateReport}
          />
        </div>
      </div>
    </div>
  );
};

export default AIResumeAnalysis;