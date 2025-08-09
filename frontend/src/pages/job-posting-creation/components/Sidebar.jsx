import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import Icon from '../../../components/AppIcon';

const Sidebar = () => {
  const location = useLocation();

  const navigationItems = [
    {
      path: '/job-posting-creation',
      icon: 'Plus',
      label: 'Criar Vaga',
      description: 'Nova oportunidade'
    },
    {
      path: '/candidate-management',
      icon: 'Users',
      label: 'Candidatos',
      description: 'Gerenciar perfis'
    },
    {
      path: '/ai-resume-analysis',
      icon: 'Brain',
      label: 'Análise IA',
      description: 'Matching inteligente'
    },
    {
      path: '/video-interview-room',
      icon: 'Video',
      label: 'Entrevistas',
      description: 'Sala virtual'
    },
    {
      path: '/interview-transcription-analysis',
      icon: 'FileText',
      label: 'Transcrições',
      description: 'Análise detalhada'
    },
    {
      path: '/recruitment-analytics-dashboard',
      icon: 'BarChart3',
      label: 'Analytics',
      description: 'Métricas e insights'
    }
  ];

  return (
    <aside className="w-64 bg-card border-r border-border h-screen sticky top-0 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <Icon name="Zap" size={20} color="white" />
          </div>
          <div>
            <h1 className="font-bold text-foreground">RecrutaAI</h1>
            <p className="text-xs text-muted-foreground">Plataforma de RH</p>
          </div>
        </div>
      </div>
      {/* Navigation */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {navigationItems?.map((item) => {
            const isActive = location?.pathname === item?.path;
            
            return (
              <Link
                key={item?.path}
                to={item?.path}
                className={`flex items-center gap-3 p-3 rounded-lg transition-smooth group ${
                  isActive
                    ? 'bg-primary text-white shadow-elevation-1'
                    : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
                }`}
              >
                <Icon 
                  name={item?.icon} 
                  size={20} 
                  className={isActive ? 'text-white' : 'text-muted-foreground group-hover:text-foreground'} 
                />
                <div className="flex-1">
                  <div className={`font-medium text-sm ${isActive ? 'text-white' : ''}`}>
                    {item?.label}
                  </div>
                  <div className={`text-xs ${isActive ? 'text-white/80' : 'text-muted-foreground'}`}>
                    {item?.description}
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </nav>
      {/* User Profile */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-3 p-3 rounded-lg hover:bg-muted/50 transition-smooth cursor-pointer">
          <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center">
            <Icon name="User" size={16} color="white" />
          </div>
          <div className="flex-1">
            <div className="font-medium text-sm text-foreground">Ana Silva</div>
            <div className="text-xs text-muted-foreground">Gerente de RH</div>
          </div>
          <Icon name="Settings" size={16} className="text-muted-foreground" />
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;