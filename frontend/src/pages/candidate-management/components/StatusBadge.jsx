import React from 'react';

const StatusBadge = ({ status }) => {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'novo':
        return {
          label: 'Novo',
          className: 'bg-blue-500/10 text-blue-400 border-blue-500/20'
        };
      case 'entrevistando':
        return {
          label: 'Entrevistando',
          className: 'bg-warning/10 text-warning border-warning/20'
        };
      case 'aprovado':
        return {
          label: 'Aprovado',
          className: 'bg-success/10 text-success border-success/20'
        };
      case 'rejeitado':
        return {
          label: 'Rejeitado',
          className: 'bg-destructive/10 text-destructive border-destructive/20'
        };
      default:
        return {
          label: 'Desconhecido',
          className: 'bg-muted/10 text-muted-foreground border-muted/20'
        };
    }
  };

  const config = getStatusConfig(status);

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${config?.className}`}>
      {config?.label}
    </span>
  );
};

export default StatusBadge;