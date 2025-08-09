import React from 'react';
import Icon from '../../../components/AppIcon';

const SoftSkillsAssessment = ({ softSkillsData }) => {
  const getSkillIcon = (skill) => {
    const iconMap = {
      'Liderança': 'Crown',
      'Trabalho em Equipe': 'Users',
      'Resolução de Problemas': 'Lightbulb',
      'Adaptabilidade': 'Shuffle',
      'Comunicação': 'MessageCircle',
      'Criatividade': 'Palette'
    };
    return iconMap?.[skill] || 'Star';
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-success';
    if (score >= 60) return 'text-warning';
    return 'text-destructive';
  };

  const getScoreBackground = (score) => {
    if (score >= 80) return 'bg-success';
    if (score >= 60) return 'bg-warning';
    return 'bg-destructive';
  };

  return (
    <div className="bg-card rounded-lg border border-border p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Avaliação de Soft Skills</h3>
      <div className="space-y-4">
        {softSkillsData?.map((skill, index) => (
          <div key={index} className="p-4 bg-muted/30 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/20 rounded-lg">
                  <Icon name={getSkillIcon(skill?.name)} size={20} className="text-primary" />
                </div>
                <div>
                  <h4 className="font-medium text-foreground">{skill?.name}</h4>
                  <p className="text-sm text-muted-foreground">{skill?.description}</p>
                </div>
              </div>
              <div className="text-right">
                <span className={`text-2xl font-bold ${getScoreColor(skill?.score)}`}>
                  {skill?.score}
                </span>
                <span className="text-sm text-muted-foreground">/100</span>
              </div>
            </div>
            
            <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-700 ${getScoreBackground(skill?.score)}`}
                style={{ width: `${skill?.score}%` }}
              />
            </div>
            
            {skill?.feedback && (
              <p className="mt-2 text-sm text-muted-foreground italic">
                "{skill?.feedback}"
              </p>
            )}
          </div>
        ))}
      </div>
      <div className="mt-6 p-4 bg-primary/10 rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <Icon name="TrendingUp" size={20} className="text-primary" />
          <span className="font-medium text-foreground">Pontuação Geral</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            Média das soft skills avaliadas
          </span>
          <span className="text-xl font-bold text-primary">
            {Math.round(softSkillsData?.reduce((acc, skill) => acc + skill?.score, 0) / softSkillsData?.length)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default SoftSkillsAssessment;