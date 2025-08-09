import React from 'react';
import Button from '../../../components/ui/Button';
import Select from '../../../components/ui/Select';

const BulkActionsBar = ({ 
  selectedCount, 
  onBulkStatusUpdate, 
  onBulkScheduleInterview, 
  onBulkExport,
  onClearSelection 
}) => {
  const statusOptions = [
    { value: 'novo', label: 'Novo' },
    { value: 'entrevistando', label: 'Entrevistando' },
    { value: 'aprovado', label: 'Aprovado' },
    { value: 'rejeitado', label: 'Rejeitado' }
  ];

  if (selectedCount === 0) return null;

  return (
    <div className="bg-primary/10 border border-primary/20 rounded-lg p-4 mb-4">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-foreground">
            {selectedCount} candidato{selectedCount > 1 ? 's' : ''} selecionado{selectedCount > 1 ? 's' : ''}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearSelection}
            className="text-muted-foreground hover:text-foreground"
          >
            Limpar seleção
          </Button>
        </div>

        <div className="flex items-center space-x-2 flex-wrap">
          <div className="flex items-center space-x-2">
            <Select
              options={statusOptions}
              placeholder="Alterar status"
              onChange={onBulkStatusUpdate}
              className="min-w-[150px]"
            />
          </div>
          
          <Button
            variant="outline"
            size="sm"
            iconName="Calendar"
            onClick={onBulkScheduleInterview}
          >
            Agendar Entrevistas
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            iconName="Download"
            onClick={onBulkExport}
          >
            Exportar
          </Button>
        </div>
      </div>
    </div>
  );
};

export default BulkActionsBar;