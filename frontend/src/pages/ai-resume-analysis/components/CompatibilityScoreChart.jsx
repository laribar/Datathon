import React, { useEffect, useState } from 'react';
import Icon from '../../../components/AppIcon';

const CompatibilityScoreChart = ({ score, breakdown }) => {
  const [animatedScore, setAnimatedScore] = useState(0);
  const [animatedBreakdown, setAnimatedBreakdown] = useState({
    technical: 0,
    experience: 0,
    education: 0
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(score);
      setAnimatedBreakdown(breakdown);
    }, 500);

    return () => clearTimeout(timer);
  }, [score, breakdown]);

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-success';
    if (score >= 60) return 'text-warning';
    return 'text-destructive';
  };

  const getScoreBackground = (score) => {
    if (score >= 80) return 'from-success/20 to-success/5';
    if (score >= 60) return 'from-warning/20 to-warning/5';
    return 'from-destructive/20 to-destructive/5';
  };

  const circumference = 2 * Math.PI * 45;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  return (
    <div className="bg-card rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Icon name="Target" size={20} className="text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Score de Compatibilidade</h3>
          <p className="text-sm text-muted-foreground">Análise IA com SBERT + LogReg</p>
        </div>
      </div>
      {/* Main Score Circle */}
      <div className="flex justify-center mb-8">
        <div className="relative">
          <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
            {/* Background circle */}
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              className="text-muted/20"
            />
            {/* Progress circle */}
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              className={`transition-all duration-1000 ease-out ${
                animatedScore >= 80 ? 'text-success' : 
                animatedScore >= 60 ? 'text-warning' : 'text-destructive'
              }`}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className={`text-3xl font-bold ${getScoreColor(animatedScore)}`}>
                {Math.round(animatedScore)}%
              </div>
              <div className="text-xs text-muted-foreground">Compatibilidade</div>
            </div>
          </div>
        </div>
      </div>
      {/* Breakdown Scores */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Icon name="Code" size={16} className="text-primary" />
            <span className="text-sm font-medium text-foreground">Habilidades Técnicas</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-1000 ease-out"
                style={{ width: `${animatedBreakdown?.technical}%` }}
              />
            </div>
            <span className="text-sm font-medium text-foreground w-8">
              {Math.round(animatedBreakdown?.technical)}%
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Icon name="Clock" size={16} className="text-accent" />
            <span className="text-sm font-medium text-foreground">Experiência</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-accent transition-all duration-1000 ease-out"
                style={{ width: `${animatedBreakdown?.experience}%` }}
              />
            </div>
            <span className="text-sm font-medium text-foreground w-8">
              {Math.round(animatedBreakdown?.experience)}%
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Icon name="GraduationCap" size={16} className="text-warning" />
            <span className="text-sm font-medium text-foreground">Educação</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-warning transition-all duration-1000 ease-out"
                style={{ width: `${animatedBreakdown?.education}%` }}
              />
            </div>
            <span className="text-sm font-medium text-foreground w-8">
              {Math.round(animatedBreakdown?.education)}%
            </span>
          </div>
        </div>
      </div>
      {/* Score Interpretation */}
      <div className={`mt-6 p-4 rounded-lg bg-gradient-to-r ${getScoreBackground(animatedScore)}`}>
        <div className="flex items-center gap-2 mb-2">
          <Icon 
            name={animatedScore >= 80 ? "CheckCircle" : animatedScore >= 60 ? "AlertCircle" : "XCircle"} 
            size={16} 
            className={getScoreColor(animatedScore)} 
          />
          <span className={`text-sm font-medium ${getScoreColor(animatedScore)}`}>
            {animatedScore >= 80 ? 'Excelente Compatibilidade' : 
             animatedScore >= 60 ? 'Boa Compatibilidade' : 'Compatibilidade Baixa'}
          </span>
        </div>
        <p className="text-xs text-muted-foreground">
          {animatedScore >= 80 
            ? 'Candidato altamente qualificado para a posição. Recomendado para entrevista.'
            : animatedScore >= 60 
            ? 'Candidato qualificado com algumas lacunas. Considere para entrevista.'
            : 'Candidato não atende aos requisitos principais. Revisar critérios.'}
        </p>
      </div>
    </div>
  );
};

export default CompatibilityScoreChart;