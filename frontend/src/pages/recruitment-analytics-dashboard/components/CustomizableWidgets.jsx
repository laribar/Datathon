import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const CustomizableWidgets = () => {
  const [widgets, setWidgets] = useState([
    { id: 1, type: 'metric', title: 'Taxa de Conversão', value: '5.8%', change: '+12%', enabled: true },
    { id: 2, type: 'metric', title: 'Tempo Médio', value: '28 dias', change: '-8%', enabled: true },
    { id: 3, type: 'chart', title: 'Fontes de Candidatos', enabled: true },
    { id: 4, type: 'list', title: 'Top Recrutadores', enabled: false },
    { id: 5, type: 'metric', title: 'Custo por Contratação', value: 'R$ 8.500', change: '-5%', enabled: true },
    { id: 6, type: 'chart', title: 'Pipeline de Candidatos', enabled: false }
  ]);

  const [isCustomizing, setIsCustomizing] = useState(false);

  const toggleWidget = (id) => {
    setWidgets(widgets?.map(widget => 
      widget?.id === id ? { ...widget, enabled: !widget?.enabled } : widget
    ));
  };

  const sourceData = [
    { name: 'LinkedIn', value: 45, color: '#0077B5' },
    { name: 'Site da Empresa', value: 28, color: '#2F9FF8' },
    { name: 'Indicações', value: 18, color: '#2ECC71' },
    { name: 'Indeed', value: 9, color: '#003A9B' }
  ];

  const topRecruiters = [
    { name: 'Ana Silva', hires: 12, avatar: 'https://randomuser.me/api/portraits/women/1.jpg' },
    { name: 'Carlos Santos', hires: 10, avatar: 'https://randomuser.me/api/portraits/men/2.jpg' },
    { name: 'Maria Oliveira', hires: 8, avatar: 'https://randomuser.me/api/portraits/women/3.jpg' }
  ];

  const renderWidget = (widget) => {
    if (!widget?.enabled) return null;

    switch (widget?.type) {
      case 'metric':
        return (
          <div key={widget?.id} className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-muted-foreground">{widget?.title}</h4>
              <div className={`flex items-center gap-1 text-xs ${
                widget?.change?.startsWith('+') ? 'text-green-400' : 'text-red-400'
              }`}>
                <Icon name={widget?.change?.startsWith('+') ? 'TrendingUp' : 'TrendingDown'} size={12} />
                {widget?.change}
              </div>
            </div>
            <p className="text-2xl font-bold text-foreground">{widget?.value}</p>
          </div>
        );

      case 'chart':
        if (widget?.title === 'Fontes de Candidatos') {
          return (
            <div key={widget?.id} className="bg-card border border-border rounded-lg p-4">
              <h4 className="text-sm font-medium text-foreground mb-4">{widget?.title}</h4>
              <div className="space-y-3">
                {sourceData?.map((source, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: source?.color }}
                      />
                      <span className="text-sm text-foreground">{source?.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-muted rounded-full h-2">
                        <div 
                          className="h-2 rounded-full transition-all duration-1000"
                          style={{ 
                            backgroundColor: source?.color,
                            width: `${source?.value}%`
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium text-foreground w-8">{source?.value}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        }
        
        if (widget?.title === 'Pipeline de Candidatos') {
          return (
            <div key={widget?.id} className="bg-card border border-border rounded-lg p-4">
              <h4 className="text-sm font-medium text-foreground mb-4">{widget?.title}</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Novos</span>
                  <span className="text-foreground font-medium">124</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Em Análise</span>
                  <span className="text-foreground font-medium">67</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Entrevistas</span>
                  <span className="text-foreground font-medium">23</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Aprovados</span>
                  <span className="text-foreground font-medium">8</span>
                </div>
              </div>
            </div>
          );
        }
        break;

      case 'list':
        if (widget?.title === 'Top Recrutadores') {
          return (
            <div key={widget?.id} className="bg-card border border-border rounded-lg p-4">
              <h4 className="text-sm font-medium text-foreground mb-4">{widget?.title}</h4>
              <div className="space-y-3">
                {topRecruiters?.map((recruiter, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <img 
                        src={recruiter?.avatar} 
                        alt={recruiter?.name}
                        className="w-8 h-8 rounded-full object-cover"
                      />
                      <span className="text-sm text-foreground">{recruiter?.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Icon name="Users" size={14} className="text-muted-foreground" />
                      <span className="text-sm font-medium text-foreground">{recruiter?.hires}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        }
        break;

      default:
        return null;
    }
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <Icon name="Layout" size={20} />
          Widgets Personalizáveis
        </h3>
        <Button
          variant={isCustomizing ? 'default' : 'outline'}
          size="sm"
          iconName="Settings"
          onClick={() => setIsCustomizing(!isCustomizing)}
        >
          {isCustomizing ? 'Concluir' : 'Personalizar'}
        </Button>
      </div>
      {isCustomizing && (
        <div className="mb-6 p-4 bg-muted/20 rounded-lg">
          <h4 className="text-sm font-medium text-foreground mb-3">Configurar Widgets</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {widgets?.map((widget) => (
              <div key={widget?.id} className="flex items-center justify-between p-3 bg-background rounded border border-border">
                <div className="flex items-center gap-3">
                  <Icon 
                    name={widget?.type === 'metric' ? 'BarChart3' : widget?.type === 'chart' ? 'PieChart' : 'List'} 
                    size={16} 
                    className="text-muted-foreground" 
                  />
                  <span className="text-sm text-foreground">{widget?.title}</span>
                </div>
                <Button
                  variant={widget?.enabled ? 'default' : 'outline'}
                  size="xs"
                  onClick={() => toggleWidget(widget?.id)}
                >
                  {widget?.enabled ? 'Ativo' : 'Inativo'}
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {widgets?.map(renderWidget)}
      </div>
      {widgets?.filter(w => w?.enabled)?.length === 0 && (
        <div className="text-center py-12">
          <Icon name="Layout" size={48} className="text-muted-foreground mx-auto mb-4" />
          <h4 className="text-lg font-medium text-foreground mb-2">Nenhum widget ativo</h4>
          <p className="text-muted-foreground mb-4">
            Clique em "Personalizar" para ativar widgets e visualizar suas métricas
          </p>
          <Button
            variant="outline"
            iconName="Settings"
            onClick={() => setIsCustomizing(true)}
          >
            Configurar Widgets
          </Button>
        </div>
      )}
    </div>
  );
};

export default CustomizableWidgets;