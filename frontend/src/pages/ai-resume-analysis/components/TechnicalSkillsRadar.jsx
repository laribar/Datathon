import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';

const TechnicalSkillsRadar = ({ skillsData }) => {
  const [hoveredSkill, setHoveredSkill] = useState(null);

  const maxScore = 100;
  const centerX = 150;
  const centerY = 150;
  const radius = 100;

  // Calculate points for radar chart
  const getPointPosition = (index, value) => {
    const angle = (index * 2 * Math.PI) / skillsData?.length - Math.PI / 2;
    const distance = (value / maxScore) * radius;
    return {
      x: centerX + Math.cos(angle) * distance,
      y: centerY + Math.sin(angle) * distance
    };
  };

  const getLabelPosition = (index) => {
    const angle = (index * 2 * Math.PI) / skillsData?.length - Math.PI / 2;
    const distance = radius + 20;
    return {
      x: centerX + Math.cos(angle) * distance,
      y: centerY + Math.sin(angle) * distance
    };
  };

  // Create path for the skill polygon
  const createPath = () => {
    return skillsData?.map((skill, index) => {
      const point = getPointPosition(index, skill?.score);
      return `${index === 0 ? 'M' : 'L'} ${point?.x} ${point?.y}`;
    })?.join(' ') + ' Z';
  };

  // Create grid circles
  const gridCircles = [20, 40, 60, 80, 100]?.map(value => (
    <circle
      key={value}
      cx={centerX}
      cy={centerY}
      r={(value / maxScore) * radius}
      fill="none"
      stroke="currentColor"
      strokeWidth="1"
      className="text-border opacity-30"
    />
  ));

  // Create grid lines
  const gridLines = skillsData?.map((_, index) => {
    const endPoint = getLabelPosition(index);
    return (
      <line
        key={index}
        x1={centerX}
        y1={centerY}
        x2={endPoint?.x - (endPoint?.x - centerX) * 0.2}
        y2={endPoint?.y - (endPoint?.y - centerY) * 0.2}
        stroke="currentColor"
        strokeWidth="1"
        className="text-border opacity-30"
      />
    );
  });

  return (
    <div className="bg-card rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-accent/10 rounded-lg">
          <Icon name="Radar" size={20} className="text-accent" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Radar de Competências</h3>
          <p className="text-sm text-muted-foreground">Análise técnica detalhada</p>
        </div>
      </div>
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Radar Chart */}
        <div className="flex-1 flex justify-center">
          <svg width="300" height="300" className="overflow-visible">
            {/* Grid */}
            {gridCircles}
            {gridLines}
            
            {/* Skill area */}
            <path
              d={createPath()}
              fill="currentColor"
              fillOpacity="0.2"
              stroke="currentColor"
              strokeWidth="2"
              className="text-primary"
            />
            
            {/* Skill points */}
            {skillsData?.map((skill, index) => {
              const point = getPointPosition(index, skill?.score);
              return (
                <circle
                  key={index}
                  cx={point?.x}
                  cy={point?.y}
                  r="4"
                  fill="currentColor"
                  className="text-primary cursor-pointer hover:text-primary/80 transition-colors"
                  onMouseEnter={() => setHoveredSkill(skill)}
                  onMouseLeave={() => setHoveredSkill(null)}
                />
              );
            })}
            
            {/* Labels */}
            {skillsData?.map((skill, index) => {
              const labelPos = getLabelPosition(index);
              return (
                <text
                  key={index}
                  x={labelPos?.x}
                  y={labelPos?.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs font-medium fill-current text-foreground"
                >
                  {skill?.name}
                </text>
              );
            })}
          </svg>
        </div>

        {/* Skills List */}
        <div className="lg:w-64 space-y-3">
          {skillsData?.map((skill, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg border transition-all cursor-pointer ${
                hoveredSkill?.name === skill?.name
                  ? 'border-primary bg-primary/5' :'border-border hover:border-primary/50'
              }`}
              onMouseEnter={() => setHoveredSkill(skill)}
              onMouseLeave={() => setHoveredSkill(null)}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-foreground">{skill?.name}</span>
                <span className="text-sm font-bold text-primary">{skill?.score}%</span>
              </div>
              <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-500 ease-out"
                  style={{ width: `${skill?.score}%` }}
                />
              </div>
              <div className="flex items-center justify-between mt-2">
                <span className="text-xs text-muted-foreground">
                  {skill?.level}
                </span>
                <div className="flex items-center gap-1">
                  {skill?.required && (
                    <Icon name="Star" size={12} className="text-warning fill-current" />
                  )}
                  {skill?.score >= 80 && (
                    <Icon name="CheckCircle" size={12} className="text-success" />
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      {/* Hover Details */}
      {hoveredSkill && (
        <div className="mt-6 p-4 bg-muted/50 rounded-lg border border-border">
          <div className="flex items-center gap-2 mb-2">
            <Icon name="Info" size={16} className="text-primary" />
            <span className="font-medium text-foreground">{hoveredSkill?.name}</span>
          </div>
          <p className="text-sm text-muted-foreground mb-2">{hoveredSkill?.description}</p>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-muted-foreground">
              Nível: <span className="text-foreground font-medium">{hoveredSkill?.level}</span>
            </span>
            <span className="text-muted-foreground">
              Experiência: <span className="text-foreground font-medium">{hoveredSkill?.experience}</span>
            </span>
            {hoveredSkill?.required && (
              <span className="text-warning font-medium">Obrigatório</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TechnicalSkillsRadar;