import React from 'react';
import Select from '../../../components/ui/Select';
import Input from '../../../components/ui/Input';
import Button from '../../../components/ui/Button';

const FilterPanel = ({ 
  filters, 
  onFilterChange, 
  onClearFilters, 
  isExpanded, 
  onToggleExpanded 
}) => {
  const statusOptions = [
    { value: '', label: 'Todos os Status' },
    { value: 'novo', label: 'Novo' },
    { value: 'entrevistando', label: 'Entrevistando' },
    { value: 'aprovado', label: 'Aprovado' },
    { value: 'rejeitado', label: 'Rejeitado' }
  ];

  const positionOptions = [
    { value: '', label: 'Todas as Posições' },
    { value: 'desenvolvedor-frontend', label: 'Desenvolvedor Frontend' },
    { value: 'desenvolvedor-backend', label: 'Desenvolvedor Backend' },
    { value: 'desenvolvedor-fullstack', label: 'Desenvolvedor Full Stack' },
    { value: 'designer-ui-ux', label: 'Designer UI/UX' },
    { value: 'gerente-produto', label: 'Gerente de Produto' },
    { value: 'analista-dados', label: 'Analista de Dados' },
    { value: 'engenheiro-devops', label: 'Engenheiro DevOps' }
  ];

  const scoreRangeOptions = [
    { value: '', label: 'Todas as Pontuações' },
    { value: '80-100', label: 'Alta Compatibilidade (80-100%)' },
    { value: '60-79', label: 'Média Compatibilidade (60-79%)' },
    { value: '0-59', label: 'Baixa Compatibilidade (0-59%)' }
  ];

  return (
    <div className="bg-card rounded-lg shadow-elevation-1 p-4 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-foreground">Filtros Avançados</h3>
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearFilters}
            className="text-muted-foreground hover:text-foreground"
          >
            Limpar Filtros
          </Button>
          <Button
            variant="ghost"
            size="sm"
            iconName={isExpanded ? "ChevronUp" : "ChevronDown"}
            onClick={onToggleExpanded}
            className="lg:hidden"
          >
            {isExpanded ? 'Ocultar' : 'Mostrar'}
          </Button>
        </div>
      </div>
      <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 ${!isExpanded ? 'hidden lg:grid' : ''}`}>
        <Select
          label="Status"
          options={statusOptions}
          value={filters?.status}
          onChange={(value) => onFilterChange('status', value)}
          placeholder="Selecionar status"
        />

        <Select
          label="Posição"
          options={positionOptions}
          value={filters?.position}
          onChange={(value) => onFilterChange('position', value)}
          placeholder="Selecionar posição"
          searchable
        />

        <Select
          label="Compatibilidade IA"
          options={scoreRangeOptions}
          value={filters?.scoreRange}
          onChange={(value) => onFilterChange('scoreRange', value)}
          placeholder="Selecionar faixa"
        />

        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Período de Aplicação</label>
          <div className="grid grid-cols-2 gap-2">
            <Input
              type="date"
              value={filters?.dateFrom}
              onChange={(e) => onFilterChange('dateFrom', e?.target?.value)}
              placeholder="Data inicial"
            />
            <Input
              type="date"
              value={filters?.dateTo}
              onChange={(e) => onFilterChange('dateTo', e?.target?.value)}
              placeholder="Data final"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default FilterPanel;