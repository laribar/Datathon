import React from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';
import CandidateScoreChart from './CandidateScoreChart';
import StatusBadge from './StatusBadge';

const CandidateModal = ({ candidate, isOpen, onClose, onScheduleInterview, onGenerateReport }) => {
  if (!isOpen || !candidate) return null;

  const skillsData = [
    { name: 'React', level: 90, color: '#61DAFB' },
    { name: 'JavaScript', level: 85, color: '#F7DF1E' },
    { name: 'Node.js', level: 75, color: '#339933' },
    { name: 'TypeScript', level: 80, color: '#3178C6' },
    { name: 'Python', level: 70, color: '#3776AB' },
    { name: 'SQL', level: 65, color: '#336791' }
  ];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card rounded-lg shadow-elevation-3 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 rounded-full overflow-hidden bg-muted">
              <Image
                src={candidate?.avatar}
                alt={candidate?.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-foreground">{candidate?.name}</h2>
              <p className="text-muted-foreground">{candidate?.email}</p>
              <p className="text-sm text-muted-foreground">{candidate?.position}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            iconName="X"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground"
          />
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Basic Info */}
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Informações Básicas</h3>
                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Status</label>
                    <div className="mt-1">
                      <StatusBadge status={candidate?.status} />
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Data de Aplicação</label>
                    <p className="text-foreground">{candidate?.applicationDate}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Departamento</label>
                    <p className="text-foreground">{candidate?.department}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Experiência</label>
                    <p className="text-foreground">{candidate?.experience || '5+ anos'}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground">Localização</label>
                    <p className="text-foreground">{candidate?.location || 'São Paulo, SP'}</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Compatibilidade IA</h3>
                <div className="flex items-center justify-center">
                  <CandidateScoreChart score={candidate?.compatibilityScore} size={120} />
                </div>
                <p className="text-center text-sm text-muted-foreground mt-2">
                  Pontuação baseada em análise de currículo e requisitos da vaga
                </p>
              </div>
            </div>

            {/* Middle Column - Skills */}
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Habilidades Técnicas</h3>
                <div className="space-y-3">
                  {skillsData?.map((skill, index) => (
                    <div key={index}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-foreground">{skill?.name}</span>
                        <span className="text-sm text-muted-foreground">{skill?.level}%</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2">
                        <div
                          className="h-2 rounded-full transition-all duration-500"
                          style={{
                            width: `${skill?.level}%`,
                            backgroundColor: skill?.color
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Resumo do Currículo</h3>
                <div className="bg-muted/20 rounded-lg p-4">
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {`Desenvolvedor Full Stack com mais de 5 anos de experiência em desenvolvimento web.\n\nEspecialista em React, Node.js e bancos de dados relacionais. Experiência comprovada em projetos de grande escala e metodologias ágeis.\n\nFormação em Ciência da Computação e certificações em AWS e Google Cloud.`}
                  </p>
                </div>
              </div>
            </div>

            {/* Right Column - Analysis */}
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Análise de Compatibilidade</h3>
                <div className="space-y-4">
                  <div className="bg-success/10 border border-success/20 rounded-lg p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Icon name="CheckCircle" size={16} className="text-success" />
                      <span className="text-sm font-medium text-success">Pontos Fortes</span>
                    </div>
                    <ul className="text-xs text-muted-foreground space-y-1">
                      <li>• Experiência sólida em React e JavaScript</li>
                      <li>• Conhecimento em arquitetura de sistemas</li>
                      <li>• Certificações relevantes</li>
                    </ul>
                  </div>

                  <div className="bg-warning/10 border border-warning/20 rounded-lg p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Icon name="AlertTriangle" size={16} className="text-warning" />
                      <span className="text-sm font-medium text-warning">Pontos de Atenção</span>
                    </div>
                    <ul className="text-xs text-muted-foreground space-y-1">
                      <li>• Pouca experiência com Docker</li>
                      <li>• Conhecimento limitado em Kubernetes</li>
                    </ul>
                  </div>

                  <div className="bg-primary/10 border border-primary/20 rounded-lg p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Icon name="Lightbulb" size={16} className="text-primary" />
                      <span className="text-sm font-medium text-primary">Recomendações</span>
                    </div>
                    <ul className="text-xs text-muted-foreground space-y-1">
                      <li>• Candidato ideal para posições React</li>
                      <li>• Considerar para entrevista técnica</li>
                      <li>• Avaliar soft skills em entrevista</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Histórico de Interações</h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                    <div>
                      <p className="text-sm text-foreground">Currículo recebido</p>
                      <p className="text-xs text-muted-foreground">08/09/2025 às 14:30</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-success rounded-full mt-2 flex-shrink-0" />
                    <div>
                      <p className="text-sm text-foreground">Análise IA concluída</p>
                      <p className="text-xs text-muted-foreground">08/09/2025 às 14:35</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end space-x-3 p-6 border-t border-border">
          <Button
            variant="outline"
            onClick={onClose}
          >
            Fechar
          </Button>
          <Button
            variant="outline"
            iconName="FileText"
            onClick={() => onGenerateReport(candidate)}
          >
            Gerar Relatório
          </Button>
          <Button
            variant="default"
            iconName="Calendar"
            onClick={() => onScheduleInterview(candidate)}
          >
            Agendar Entrevista
          </Button>
        </div>
      </div>
    </div>
  );
};

export default CandidateModal;