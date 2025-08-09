import React from 'react';
import Icon from '../../../components/AppIcon';

const JobRequirementsPanel = ({ jobData, matchedKeywords, missingKeywords }) => {
  return (
    <div className="bg-card rounded-lg p-6 h-full">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Icon name="FileText" size={20} className="text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Requisitos da Vaga</h3>
          <p className="text-sm text-muted-foreground">{jobData?.title}</p>
        </div>
      </div>
      <div className="space-y-6">
        {/* Job Description */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Descrição</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {jobData?.description}
          </p>
        </div>

        {/* Required Skills */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Habilidades Técnicas</h4>
          <div className="flex flex-wrap gap-2">
            {jobData?.requiredSkills?.map((skill, index) => {
              const isMatched = matchedKeywords?.includes(skill?.toLowerCase());
              const isMissing = missingKeywords?.includes(skill?.toLowerCase());
              
              return (
                <span
                  key={index}
                  className={`px-3 py-1 rounded-full text-xs font-medium ${
                    isMatched
                      ? 'bg-success/20 text-success border border-success/30'
                      : isMissing
                      ? 'bg-destructive/20 text-destructive border border-destructive/30' :'bg-muted text-muted-foreground border border-border'
                  }`}
                >
                  {skill}
                  {isMatched && <Icon name="Check" size={12} className="inline ml-1" />}
                  {isMissing && <Icon name="X" size={12} className="inline ml-1" />}
                </span>
              );
            })}
          </div>
        </div>

        {/* Experience Level */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Experiência</h4>
          <div className="flex items-center gap-2">
            <Icon name="Clock" size={16} className="text-muted-foreground" />
            <span className="text-sm text-muted-foreground">{jobData?.experienceLevel}</span>
          </div>
        </div>

        {/* Education */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Educação</h4>
          <div className="flex items-center gap-2">
            <Icon name="GraduationCap" size={16} className="text-muted-foreground" />
            <span className="text-sm text-muted-foreground">{jobData?.education}</span>
          </div>
        </div>

        {/* Location */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Localização</h4>
          <div className="flex items-center gap-2">
            <Icon name="MapPin" size={16} className="text-muted-foreground" />
            <span className="text-sm text-muted-foreground">{jobData?.location}</span>
          </div>
        </div>

        {/* Salary Range */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Faixa Salarial</h4>
          <div className="flex items-center gap-2">
            <Icon name="DollarSign" size={16} className="text-muted-foreground" />
            <span className="text-sm text-muted-foreground">{jobData?.salaryRange}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default JobRequirementsPanel;