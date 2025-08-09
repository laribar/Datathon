import React from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';

const CandidateQualificationsPanel = ({ candidateData, matchedKeywords, missingKeywords }) => {
  return (
    <div className="bg-card rounded-lg p-6 h-full">
      <div className="flex items-center gap-3 mb-6">
        <Image
          src={candidateData?.avatar}
          alt={candidateData?.name}
          className="w-12 h-12 rounded-full object-cover"
        />
        <div>
          <h3 className="text-lg font-semibold text-foreground">{candidateData?.name}</h3>
          <p className="text-sm text-muted-foreground">{candidateData?.position}</p>
        </div>
      </div>
      <div className="space-y-6">
        {/* Summary */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Resumo Profissional</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {candidateData?.summary}
          </p>
        </div>

        {/* Skills */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Habilidades</h4>
          <div className="flex flex-wrap gap-2">
            {candidateData?.skills?.map((skill, index) => {
              const isMatched = matchedKeywords?.includes(skill?.toLowerCase());
              const isMissing = missingKeywords?.includes(skill?.toLowerCase());
              
              return (
                <span
                  key={index}
                  className={`px-3 py-1 rounded-full text-xs font-medium ${
                    isMatched
                      ? 'bg-success/20 text-success border border-success/30' :'bg-muted text-muted-foreground border border-border'
                  }`}
                >
                  {skill}
                  {isMatched && <Icon name="Check" size={12} className="inline ml-1" />}
                </span>
              );
            })}
          </div>
        </div>

        {/* Experience */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Experiência</h4>
          <div className="space-y-4">
            {candidateData?.experience?.map((exp, index) => (
              <div key={index} className="border-l-2 border-primary/30 pl-4">
                <h5 className="font-medium text-foreground text-sm">{exp?.position}</h5>
                <p className="text-xs text-muted-foreground">{exp?.company} • {exp?.duration}</p>
                <p className="text-xs text-muted-foreground mt-1">{exp?.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Education */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Educação</h4>
          <div className="space-y-3">
            {candidateData?.education?.map((edu, index) => (
              <div key={index} className="flex items-start gap-2">
                <Icon name="GraduationCap" size={16} className="text-muted-foreground mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-foreground">{edu?.degree}</p>
                  <p className="text-xs text-muted-foreground">{edu?.institution} • {edu?.year}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Contact Info */}
        <div>
          <h4 className="font-medium text-foreground mb-3">Contato</h4>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Icon name="Mail" size={14} className="text-muted-foreground" />
              <span className="text-sm text-muted-foreground">{candidateData?.email}</span>
            </div>
            <div className="flex items-center gap-2">
              <Icon name="Phone" size={14} className="text-muted-foreground" />
              <span className="text-sm text-muted-foreground">{candidateData?.phone}</span>
            </div>
            <div className="flex items-center gap-2">
              <Icon name="MapPin" size={14} className="text-muted-foreground" />
              <span className="text-sm text-muted-foreground">{candidateData?.location}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CandidateQualificationsPanel;