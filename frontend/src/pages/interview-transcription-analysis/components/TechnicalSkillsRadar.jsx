import React from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

const TechnicalSkillsRadar = ({ skillsData }) => {
  return (
    <div className="bg-card rounded-lg border border-border p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Habilidades Técnicas</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={skillsData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
            <PolarGrid stroke="rgba(148, 163, 184, 0.2)" />
            <PolarAngleAxis 
              dataKey="skill" 
              tick={{ fill: '#94A3B8', fontSize: 11 }}
              className="text-xs"
            />
            <PolarRadiusAxis 
              angle={90} 
              domain={[0, 100]} 
              tick={{ fill: '#94A3B8', fontSize: 10 }}
            />
            <Radar
              name="Proficiência"
              dataKey="proficiency"
              stroke="#2F9FF8"
              fill="#2F9FF8"
              fillOpacity={0.2}
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 space-y-2">
        {skillsData?.map((skill, index) => (
          <div key={index} className="flex items-center justify-between">
            <span className="text-sm text-foreground">{skill?.skill}</span>
            <div className="flex items-center gap-2">
              <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary rounded-full transition-all duration-500"
                  style={{ width: `${skill?.proficiency}%` }}
                />
              </div>
              <span className="text-sm font-medium text-primary min-w-[40px]">
                {skill?.proficiency}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TechnicalSkillsRadar;