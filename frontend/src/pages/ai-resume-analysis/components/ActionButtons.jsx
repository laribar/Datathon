import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const ActionButtons = ({ candidateData, onStatusChange, onScheduleInterview, onGenerateReport }) => {
  const [isLoading, setIsLoading] = useState({
    approve: false,
    reject: false,
    schedule: false,
    report: false
  });

  const handleAction = async (action, callback) => {
    setIsLoading(prev => ({ ...prev, [action]: true }));
    
    // Simulate API call
    setTimeout(() => {
      callback();
      setIsLoading(prev => ({ ...prev, [action]: false }));
    }, 1500);
  };

  const pipelineStages = [
    { id: 'new', label: 'Novo', color: 'bg-muted text-muted-foreground' },
    { id: 'screening', label: 'Triagem', color: 'bg-primary text-primary-foreground' },
    { id: 'interviewing', label: 'Entrevista', color: 'bg-warning text-warning-foreground' },
    { id: 'approved', label: 'Aprovado', color: 'bg-success text-success-foreground' },
    { id: 'rejected', label: 'Rejeitado', color: 'bg-destructive text-destructive-foreground' }
  ];

  const currentStageIndex = pipelineStages?.findIndex(stage => stage?.id === candidateData?.status);

  return (
    <div className="bg-card rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Icon name="Settings" size={20} className="text-primary" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Ações do Candidato</h3>
          <p className="text-sm text-muted-foreground">Gerenciar pipeline de recrutamento</p>
        </div>
      </div>
      {/* Pipeline Status */}
      <div className="mb-6">
        <h4 className="font-medium text-foreground mb-3">Status no Pipeline</h4>
        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          {pipelineStages?.map((stage, index) => (
            <div key={stage?.id} className="flex items-center gap-2 flex-shrink-0">
              <div
                className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${
                  index <= currentStageIndex
                    ? stage?.color
                    : 'bg-muted/50 text-muted-foreground'
                }`}
              >
                {stage?.label}
              </div>
              {index < pipelineStages?.length - 1 && (
                <Icon
                  name="ChevronRight"
                  size={14}
                  className={`${
                    index < currentStageIndex
                      ? 'text-primary' :'text-muted-foreground'
                  }`}
                />
              )}
            </div>
          ))}
        </div>
      </div>
      {/* Primary Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <Button
          variant="success"
          fullWidth
          loading={isLoading?.approve}
          iconName="CheckCircle"
          iconPosition="left"
          onClick={() => handleAction('approve', () => onStatusChange('approved'))}
        >
          Aprovar Candidato
        </Button>

        <Button
          variant="destructive"
          fullWidth
          loading={isLoading?.reject}
          iconName="XCircle"
          iconPosition="left"
          onClick={() => handleAction('reject', () => onStatusChange('rejected'))}
        >
          Rejeitar Candidato
        </Button>
      </div>
      {/* Secondary Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
        <Button
          variant="outline"
          fullWidth
          loading={isLoading?.schedule}
          iconName="Calendar"
          iconPosition="left"
          onClick={() => handleAction('schedule', onScheduleInterview)}
        >
          Agendar Entrevista
        </Button>

        <Button
          variant="outline"
          fullWidth
          loading={isLoading?.report}
          iconName="FileText"
          iconPosition="left"
          onClick={() => handleAction('report', onGenerateReport)}
        >
          Gerar Relatório
        </Button>
      </div>
      {/* Additional Actions */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            iconName="Share"
            iconPosition="left"
          >
            Compartilhar Perfil
          </Button>
          <Button
            variant="ghost"
            size="sm"
            iconName="MessageSquare"
            iconPosition="left"
          >
            Enviar Mensagem
          </Button>
          <Button
            variant="ghost"
            size="sm"
            iconName="Phone"
            iconPosition="left"
          >
            Ligar
          </Button>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            iconName="Star"
            iconPosition="left"
          >
            Adicionar aos Favoritos
          </Button>
          <Button
            variant="ghost"
            size="sm"
            iconName="Tag"
            iconPosition="left"
          >
            Adicionar Tags
          </Button>
          <Button
            variant="ghost"
            size="sm"
            iconName="Archive"
            iconPosition="left"
          >
            Arquivar
          </Button>
        </div>
      </div>
      {/* Quick Notes */}
      <div className="mt-6 pt-6 border-t border-border">
        <h4 className="font-medium text-foreground mb-3">Notas Rápidas</h4>
        <textarea
          placeholder="Adicione suas observações sobre o candidato..."
          className="w-full h-20 p-3 bg-muted/30 border border-border rounded-lg text-sm text-foreground placeholder-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
        />
        <div className="flex justify-end mt-2">
          <Button variant="outline" size="sm">
            Salvar Nota
          </Button>
        </div>
      </div>
    </div>
  );
};

export default ActionButtons;