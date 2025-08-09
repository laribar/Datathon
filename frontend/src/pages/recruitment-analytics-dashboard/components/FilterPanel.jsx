import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';
import Select from '../../../components/ui/Select';

const FilterPanel = ({ onFiltersChange, isCollapsed, onToggle }) => {
  const [filters, setFilters] = useState({
    dateRange: '30d',
    department: 'all',
    positionType: 'all',
    hiringManager: 'all'
  });

  const dateRangeOptions = [
    { value: '7d', label: 'Últimos 7 dias' },
    { value: '30d', label: 'Últimos 30 dias' },
    { value: '90d', label: 'Últimos 90 dias' },
    { value: '1y', label: 'Último ano' },
    { value: 'custom', label: 'Período personalizado' }
  ];

  const departmentOptions = [
    { value: 'all', label: 'Todos os departamentos' },
    { value: 'engineering', label: 'Engenharia' },
    { value: 'marketing', label: 'Marketing' },
    { value: 'sales', label: 'Vendas' },
    { value: 'hr', label: 'Recursos Humanos' },
    { value: 'finance', label: 'Financeiro' }
  ];

  const positionTypeOptions = [
    { value: 'all', label: 'Todos os tipos' },
    { value: 'full-time', label: 'Tempo integral' },
    { value: 'part-time', label: 'Meio período' },
    { value: 'contract', label: 'Contrato' },
    { value: 'internship', label: 'Estágio' }
  ];

  const hiringManagerOptions = [
    { value: 'all', label: 'Todos os gerentes' },
    { value: 'ana-silva', label: 'Ana Silva' },
    { value: 'carlos-santos', label: 'Carlos Santos' },
    { value: 'maria-oliveira', label: 'Maria Oliveira' },
    { value: 'joao-ferreira', label: 'João Ferreira' }
  ];

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    onFiltersChange(newFilters);
  };

  const resetFilters = () => {
    const defaultFilters = {
      dateRange: '30d',
      department: 'all',
      positionType: 'all',
      hiringManager: 'all'
    };
    setFilters(defaultFilters);
    onFiltersChange(defaultFilters);
  };

  return (
    <div className={`bg-card border border-border rounded-lg transition-all duration-300 ${
      isCollapsed ? 'p-4' : 'p-6'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <Icon name="Filter" size={20} />
          Filtros Avançados
        </h3>
        <Button
          variant="ghost"
          size="sm"
          iconName={isCollapsed ? "ChevronDown" : "ChevronUp"}
          onClick={onToggle}
        >
          {isCollapsed ? 'Expandir' : 'Recolher'}
        </Button>
      </div>
      {!isCollapsed && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Select
              label="Período"
              options={dateRangeOptions}
              value={filters?.dateRange}
              onChange={(value) => handleFilterChange('dateRange', value)}
            />

            <Select
              label="Departamento"
              options={departmentOptions}
              value={filters?.department}
              onChange={(value) => handleFilterChange('department', value)}
            />

            <Select
              label="Tipo de Posição"
              options={positionTypeOptions}
              value={filters?.positionType}
              onChange={(value) => handleFilterChange('positionType', value)}
            />

            <Select
              label="Gerente de Contratação"
              options={hiringManagerOptions}
              value={filters?.hiringManager}
              onChange={(value) => handleFilterChange('hiringManager', value)}
            />
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-border">
            <Button
              variant="outline"
              size="sm"
              iconName="RotateCcw"
              onClick={resetFilters}
            >
              Limpar Filtros
            </Button>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                iconName="Download"
              >
                Exportar CSV
              </Button>
              <Button
                variant="outline"
                size="sm"
                iconName="FileText"
              >
                Exportar PDF
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FilterPanel;