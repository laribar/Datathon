import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';

// Import all components
import KPICard from './components/KPICard';
import FilterPanel from './components/FilterPanel';
import HiringFunnelChart from './components/HiringFunnelChart';
import MonthlyVolumeChart from './components/MonthlyVolumeChart';
import DepartmentChart from './components/DepartmentChart';
import PredictiveAnalytics from './components/PredictiveAnalytics';
import ComparisonAnalysis from './components/ComparisonAnalysis';
import CustomizableWidgets from './components/CustomizableWidgets';

const RecruitmentAnalyticsDashboard = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [filtersCollapsed, setFiltersCollapsed] = useState(false);
  const [currentFilters, setCurrentFilters] = useState({});

  // Mock KPI data
  const kpiData = [
    {
      title: 'Tempo Médio de Contratação',
      value: '28 dias',
      change: '-8%',
      changeType: 'down',
      icon: 'Clock',
      color: 'blue'
    },
    {
      title: 'Custo por Contratação',
      value: 'R$ 8.500',
      change: '-5%',
      changeType: 'down',
      icon: 'DollarSign',
      color: 'green'
    },
    {
      title: 'Taxa de Conversão',
      value: '5.8%',
      change: '+12%',
      changeType: 'up',
      icon: 'Target',
      color: 'purple'
    },
    {
      title: 'Satisfação do Candidato',
      value: '4.6/5',
      change: '+3%',
      changeType: 'up',
      icon: 'Star',
      color: 'orange'
    }
  ];

  const navigationRoutes = [
    { path: '/job-posting-creation', label: 'Criar Vaga', icon: 'Plus' },
    { path: '/candidate-management', label: 'Candidatos', icon: 'Users' },
    { path: '/ai-resume-analysis', label: 'Análise IA', icon: 'Brain' },
    { path: '/video-interview-room', label: 'Entrevistas', icon: 'Video' },
    { path: '/interview-transcription-analysis', label: 'Transcrições', icon: 'FileText' },
    { path: '/recruitment-analytics-dashboard', label: 'Analytics', icon: 'BarChart3' }
  ];

  const handleFiltersChange = (filters) => {
    setCurrentFilters(filters);
    // Here you would typically update your data based on filters
    console.log('Filters updated:', filters);
  };

  useEffect(() => {
    // Set page title
    document.title = 'Analytics Dashboard - Plataforma de Recrutamento IA';
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              iconName="Menu"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            />
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Icon name="Brain" size={20} className="text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">Plataforma de Recrutamento IA</h1>
                <p className="text-sm text-muted-foreground">Dashboard de Analytics</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Button variant="outline" size="sm" iconName="Download">
              Exportar Relatório
            </Button>
            <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center">
              <Icon name="User" size={16} className="text-muted-foreground" />
            </div>
          </div>
        </div>
      </header>
      <div className="flex">
        {/* Sidebar */}
        <aside className={`bg-card border-r border-border transition-all duration-300 ${
          sidebarCollapsed ? 'w-16' : 'w-64'
        }`}>
          <nav className="p-4">
            <div className="space-y-2">
              {navigationRoutes?.map((route) => (
                <Link
                  key={route?.path}
                  to={route?.path}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                    route?.path === '/recruitment-analytics-dashboard' ?'bg-primary text-white' :'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                  }`}
                >
                  <Icon name={route?.icon} size={20} />
                  {!sidebarCollapsed && <span className="text-sm font-medium">{route?.label}</span>}
                </Link>
              ))}
            </div>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Page Header */}
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-foreground">Dashboard de Analytics</h2>
                <p className="text-muted-foreground">
                  Insights abrangentes de performance de contratação e análise preditiva
                </p>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Icon name="Calendar" size={16} />
                <span>Atualizado em {new Date()?.toLocaleDateString('pt-BR')}</span>
              </div>
            </div>

            {/* Filter Panel */}
            <FilterPanel
              onFiltersChange={handleFiltersChange}
              isCollapsed={filtersCollapsed}
              onToggle={() => setFiltersCollapsed(!filtersCollapsed)}
            />

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {kpiData?.map((kpi, index) => (
                <KPICard 
                  key={index} 
                  title={kpi.title}
                  value={kpi.value}
                  change={kpi.change}
                  changeType={kpi.changeType}
                  icon={kpi.icon}
                  color={kpi.color}
                />
              ))}
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HiringFunnelChart />
              <DepartmentChart />
            </div>

            {/* Monthly Volume Chart */}
            <MonthlyVolumeChart />

            {/* Analysis Sections */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <PredictiveAnalytics />
              <ComparisonAnalysis />
            </div>

            {/* Customizable Widgets */}
            <CustomizableWidgets />

            {/* Quick Actions */}
            <div className="bg-card border border-border rounded-lg p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
                <Icon name="Zap" size={20} />
                Ações Rápidas
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Link to="/job-posting-creation">
                  <Button variant="outline" fullWidth iconName="Plus" iconPosition="left">
                    Nova Vaga
                  </Button>
                </Link>
                <Link to="/candidate-management">
                  <Button variant="outline" fullWidth iconName="Users" iconPosition="left">
                    Ver Candidatos
                  </Button>
                </Link>
                <Link to="/ai-resume-analysis">
                  <Button variant="outline" fullWidth iconName="Brain" iconPosition="left">
                    Análise IA
                  </Button>
                </Link>
                <Link to="/video-interview-room">
                  <Button variant="outline" fullWidth iconName="Video" iconPosition="left">
                    Entrevistas
                  </Button>
                </Link>
              </div>
            </div>

            {/* Footer */}
            <footer className="text-center py-6 border-t border-border">
              <p className="text-sm text-muted-foreground">
                © {new Date()?.getFullYear()} Plataforma de Recrutamento IA. Todos os direitos reservados.
              </p>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
};

export default RecruitmentAnalyticsDashboard;