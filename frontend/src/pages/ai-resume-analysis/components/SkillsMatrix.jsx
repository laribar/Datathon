import React from 'react';
import Icon from '../../../components/AppIcon';

const SkillsMatrix = ({ requiredSkills, candidateSkills }) => {
  // Combine and analyze skills
  const skillsAnalysis = requiredSkills?.map(reqSkill => {
    const candidateSkill = candidateSkills?.find(
      cs => cs?.name?.toLowerCase() === reqSkill?.name?.toLowerCase()
    );
    
    return {
      name: reqSkill?.name,
      required: reqSkill?.level,
      candidate: candidateSkill ? candidateSkill?.level : 0,
      proficiency: candidateSkill ? candidateSkill?.proficiency : 'Não informado',
      match: candidateSkill ? candidateSkill?.level >= reqSkill?.level : false,
      gap: candidateSkill ? Math.max(0, reqSkill?.level - candidateSkill?.level) : reqSkill?.level
    };
  });

  const getSkillColor = (match, gap) => {
    if (match) return 'text-success';
    if (gap <= 20) return 'text-warning';
    return 'text-destructive';
  };

  const getProgressColor = (match, gap) => {
    if (match) return 'bg-success';
    if (gap <= 20) return 'bg-warning';
    return 'bg-destructive';
  };

  const getProficiencyIcon = (proficiency) => {
    switch (proficiency?.toLowerCase()) {
      case 'iniciante':
        return 'Circle';
      case 'intermediário':
        return 'CircleDot';
      case 'avançado':
        return 'CheckCircle';
      case 'expert':
        return 'Star';
      default:
        return 'HelpCircle';
    }
  };

  return (
    <div className="bg-card rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-warning/10 rounded-lg">
          <Icon name="BarChart3" size={20} className="text-warning" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Matriz de Habilidades</h3>
          <p className="text-sm text-muted-foreground">Comparação detalhada de competências</p>
        </div>
      </div>
      {/* Skills Comparison */}
      <div className="space-y-4 mb-6">
        {skillsAnalysis?.map((skill, index) => (
          <div key={index} className="border border-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="font-medium text-foreground">{skill?.name}</span>
                <Icon 
                  name={getProficiencyIcon(skill?.proficiency)} 
                  size={14} 
                  className={getSkillColor(skill?.match, skill?.gap)} 
                />
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-sm font-medium ${getSkillColor(skill?.match, skill?.gap)}`}>
                  {skill?.match ? 'Atende' : `Gap: ${skill?.gap}%`}
                </span>
                <Icon 
                  name={skill?.match ? "CheckCircle" : skill?.gap <= 20 ? "AlertCircle" : "XCircle"} 
                  size={16} 
                  className={getSkillColor(skill?.match, skill?.gap)} 
                />
              </div>
            </div>

            {/* Progress Bars */}
            <div className="space-y-2">
              {/* Required Level */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Nível Requerido</span>
                  <span className="text-xs font-medium text-foreground">{skill?.required}%</span>
                </div>
                <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-500"
                    style={{ width: `${skill?.required}%` }}
                  />
                </div>
              </div>

              {/* Candidate Level */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Nível do Candidato</span>
                  <span className="text-xs font-medium text-foreground">{skill?.candidate}%</span>
                </div>
                <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${getProgressColor(skill?.match, skill?.gap)}`}
                    style={{ width: `${skill?.candidate}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Proficiency Level */}
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-border">
              <span className="text-xs text-muted-foreground">Proficiência</span>
              <span className="text-xs font-medium text-foreground">{skill?.proficiency}</span>
            </div>
          </div>
        ))}
      </div>
      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-muted/30 rounded-lg">
        <div className="text-center">
          <div className="text-2xl font-bold text-success mb-1">
            {skillsAnalysis?.filter(s => s?.match)?.length}
          </div>
          <div className="text-xs text-muted-foreground">Habilidades Atendidas</div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-warning mb-1">
            {skillsAnalysis?.filter(s => !s?.match && s?.gap <= 20)?.length}
          </div>
          <div className="text-xs text-muted-foreground">Gaps Menores</div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-destructive mb-1">
            {skillsAnalysis?.filter(s => s?.gap > 20)?.length}
          </div>
          <div className="text-xs text-muted-foreground">Gaps Críticos</div>
        </div>
      </div>
    </div>
  );
};

export default SkillsMatrix;