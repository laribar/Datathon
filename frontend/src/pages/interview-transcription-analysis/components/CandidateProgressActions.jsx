import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const CandidateProgressActions = ({ candidate, onStatusChange, onGenerateReport }) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [selectedAction, setSelectedAction] = useState(null);

  const statusOptions = [
    {
      id: 'approved',
      label: 'Aprovar Candidato',
      icon: 'CheckCircle',
      color: 'success',
      description: 'Avançar para próxima etapa do processo'
    },
    {
      id: 'rejected',
      label: 'Reprovar Candidato',
      icon: 'XCircle',
      color: 'destructive',
      description: 'Finalizar processo para este candidato'
    },
    {
      id: 'pending',
      label: 'Aguardar Decisão',
      icon: 'Clock',
      color: 'warning',
      description: 'Manter em análise para revisão posterior'
    },
    {
      id: 'interview_again',
      label: 'Nova Entrevista',
      icon: 'RotateCcw',
      color: 'primary',
      description: 'Agendar entrevista adicional'
    }
  ];

  const handleActionClick = (action) => {
    setSelectedAction(action);
    setShowConfirmation(true);
  };

  const handleConfirmAction = () => {
    onStatusChange(candidate?.id, selectedAction?.id);
    setShowConfirmation(false);
    setSelectedAction(null);
  };

  const handleCancelAction = () => {
    setShowConfirmation(false);
    setSelectedAction(null);
  };

  return (
    <div className="bg-card rounded-lg border border-border p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground">Ações do Candidato</h3>
        <div className="flex items-center gap-2">
          <Icon name="User" size={20} className="text-muted-foreground" />
          <span className="text-sm text-muted-foreground">{candidate?.name}</span>
        </div>
      </div>
      {/* Current Status */}
      <div className="mb-6 p-4 bg-muted/30 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm text-muted-foreground">Status Atual</span>
            <div className="flex items-center gap-2 mt-1">
              <div className={`w-3 h-3 rounded-full ${
                candidate?.status === 'approved' ? 'bg-success' :
                candidate?.status === 'rejected' ? 'bg-destructive' :
                candidate?.status === 'pending' ? 'bg-warning' : 'bg-primary'
              }`} />
              <span className="font-medium text-foreground">
                {candidate?.status === 'approved' ? 'Aprovado' :
                 candidate?.status === 'rejected' ? 'Reprovado' :
                 candidate?.status === 'pending' ? 'Pendente' : 'Em Análise'}
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="text-sm text-muted-foreground">Pontuação Geral</span>
            <div className="text-2xl font-bold text-primary">{candidate?.overallScore}/100</div>
          </div>
        </div>
      </div>
      {/* Action Buttons */}
      <div className="space-y-3 mb-6">
        {statusOptions?.map((option) => (
          <button
            key={option?.id}
            onClick={() => handleActionClick(option)}
            className={`w-full p-4 rounded-lg border-2 border-dashed transition-all hover:border-solid ${
              option?.color === 'success' ? 'border-success/30 hover:border-success hover:bg-success/10' :
              option?.color === 'destructive' ? 'border-destructive/30 hover:border-destructive hover:bg-destructive/10' :
              option?.color === 'warning'? 'border-warning/30 hover:border-warning hover:bg-warning/10' : 'border-primary/30 hover:border-primary hover:bg-primary/10'
            }`}
          >
            <div className="flex items-center gap-3">
              <Icon 
                name={option?.icon} 
                size={20} 
                className={
                  option?.color === 'success' ? 'text-success' :
                  option?.color === 'destructive' ? 'text-destructive' :
                  option?.color === 'warning'? 'text-warning' : 'text-primary'
                }
              />
              <div className="text-left">
                <div className="font-medium text-foreground">{option?.label}</div>
                <div className="text-sm text-muted-foreground">{option?.description}</div>
              </div>
            </div>
          </button>
        ))}
      </div>
      {/* Report Generation */}
      <div className="border-t border-border pt-6">
        <h4 className="font-medium text-foreground mb-3">Relatórios e Exportação</h4>
        <div className="flex flex-col gap-2">
          <Button
            variant="outline"
            onClick={() => onGenerateReport('comprehensive')}
            iconName="FileText"
            fullWidth
          >
            Gerar Relatório Completo
          </Button>
          <Button
            variant="outline"
            onClick={() => onGenerateReport('summary')}
            iconName="Download"
            fullWidth
          >
            Exportar Resumo Executivo
          </Button>
          <Button
            variant="outline"
            onClick={() => onGenerateReport('share')}
            iconName="Share2"
            fullWidth
          >
            Compartilhar Perfil
          </Button>
        </div>
      </div>
      {/* Confirmation Modal */}
      {showConfirmation && selectedAction && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card rounded-lg border border-border p-6 max-w-md w-full mx-4">
            <div className="flex items-center gap-3 mb-4">
              <Icon 
                name={selectedAction?.icon} 
                size={24} 
                className={
                  selectedAction?.color === 'success' ? 'text-success' :
                  selectedAction?.color === 'destructive' ? 'text-destructive' :
                  selectedAction?.color === 'warning'? 'text-warning' : 'text-primary'
                }
              />
              <h3 className="text-lg font-semibold text-foreground">
                Confirmar Ação
              </h3>
            </div>
            
            <p className="text-foreground mb-2">
              Tem certeza que deseja <strong>{selectedAction?.label?.toLowerCase()}</strong>?
            </p>
            <p className="text-sm text-muted-foreground mb-6">
              {selectedAction?.description}
            </p>
            
            <div className="flex gap-3">
              <Button
                variant={selectedAction?.color === 'destructive' ? 'destructive' : 'default'}
                onClick={handleConfirmAction}
                fullWidth
              >
                Confirmar
              </Button>
              <Button
                variant="outline"
                onClick={handleCancelAction}
                fullWidth
              >
                Cancelar
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CandidateProgressActions;