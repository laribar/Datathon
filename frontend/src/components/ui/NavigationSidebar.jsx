import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Icon from '../AppIcon';

const NavigationSidebar = ({ isCollapsed = false, onToggle }) => {
  const [isExpanded, setIsExpanded] = useState(!isCollapsed);
  const [interviewMode, setInterviewMode] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const navigationItems = [
    {
      label: 'Vagas',
      path: '/job-posting-creation',
      icon: 'Briefcase',
      tooltip: 'Criação e gerenciamento de vagas'
    },
    {
      label: 'Candidatos',
      path: '/candidate-management',
      icon: 'Users',
      tooltip: 'Gestão completa de candidatos',
      subItems: [
        {
          label: 'Análise de CV',
          path: '/ai-resume-analysis',
          icon: 'FileText'
        }
      ]
    },
    {
      label: 'Entrevistas',
      path: '/video-interview-room',
      icon: 'Video',
      tooltip: 'Sala de entrevistas e análises',
      subItems: [
        {
          label: 'Transcrições',
          path: '/interview-transcription-analysis',
          icon: 'MessageSquare'
        }
      ]
    },
    {
      label: 'Analytics',
      path: '/recruitment-analytics-dashboard',
      icon: 'BarChart3',
      tooltip: 'Dashboard de métricas de recrutamento'
    }
  ];

  useEffect(() => {
    const isInterviewRoom = location?.pathname === '/video-interview-room';
    setInterviewMode(isInterviewRoom);
    
    if (isInterviewRoom && isExpanded) {
      setIsExpanded(false);
    }
  }, [location?.pathname, isExpanded]);

  useEffect(() => {
    setIsExpanded(!isCollapsed);
  }, [isCollapsed]);

  const handleNavigation = (path) => {
    navigate(path);
  };

  const toggleSidebar = () => {
    const newState = !isExpanded;
    setIsExpanded(newState);
    if (onToggle) {
      onToggle(!newState);
    }
  };

  const isActiveRoute = (path) => {
    return location?.pathname === path;
  };

  const hasActiveSubItem = (item) => {
    if (!item?.subItems) return false;
    return item?.subItems?.some(subItem => isActiveRoute(subItem?.path));
  };

  if (interviewMode) {
    return (
      <div className="fixed top-0 left-0 right-0 z-300 bg-card border-b border-border">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Icon name="Zap" size={20} color="white" />
              </div>
              <span className="text-lg font-semibold text-foreground">AI Recruitment</span>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-muted-foreground">Modo Entrevista Ativo</span>
            <button
              onClick={() => navigate('/candidate-management')}
              className="px-4 py-2 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 transition-smooth"
            >
              Sair da Entrevista
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* Desktop Sidebar */}
      <div className={`fixed left-0 top-0 h-full bg-card border-r border-border z-100 transition-all duration-300 ease-out ${
        isExpanded ? 'w-60' : 'w-16'
      } hidden md:block`}>
        
        {/* Logo Section */}
        <div className="flex items-center px-4 py-6 border-b border-border">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
              <Icon name="Zap" size={20} color="white" />
            </div>
            {isExpanded && (
              <div className="overflow-hidden">
                <h1 className="text-lg font-semibold text-foreground whitespace-nowrap">
                  AI Recruitment
                </h1>
                <p className="text-xs text-muted-foreground whitespace-nowrap">
                  Platform
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigationItems?.map((item) => {
            const isActive = isActiveRoute(item?.path) || hasActiveSubItem(item);
            
            return (
              <div key={item?.path}>
                <button
                  onClick={() => handleNavigation(item?.path)}
                  className={`w-full flex items-center px-3 py-3 rounded-lg transition-smooth group relative ${
                    isActive 
                      ? 'bg-primary text-primary-foreground shadow-elevation-1' 
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                  title={!isExpanded ? item?.tooltip : ''}
                >
                  <Icon 
                    name={item?.icon} 
                    size={20} 
                    className="flex-shrink-0"
                  />
                  {isExpanded && (
                    <span className="ml-3 font-medium text-sm">
                      {item?.label}
                    </span>
                  )}
                  
                  {!isExpanded && (
                    <div className="absolute left-full ml-2 px-2 py-1 bg-popover text-popover-foreground text-xs rounded shadow-elevation-2 opacity-0 group-hover:opacity-100 transition-smooth pointer-events-none whitespace-nowrap z-200">
                      {item?.tooltip}
                    </div>
                  )}
                </button>
                {/* Sub Items */}
                {item?.subItems && isExpanded && (
                  <div className="ml-6 mt-1 space-y-1">
                    {item?.subItems?.map((subItem) => (
                      <button
                        key={subItem?.path}
                        onClick={() => handleNavigation(subItem?.path)}
                        className={`w-full flex items-center px-3 py-2 rounded-md transition-smooth text-sm ${
                          isActiveRoute(subItem?.path)
                            ? 'bg-accent text-accent-foreground'
                            : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                        }`}
                      >
                        <Icon name={subItem?.icon} size={16} />
                        <span className="ml-2">{subItem?.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        {/* Toggle Button */}
        <div className="px-4 py-4 border-t border-border">
          <button
            onClick={toggleSidebar}
            className="w-full flex items-center justify-center px-3 py-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-smooth"
          >
            <Icon 
              name={isExpanded ? "ChevronLeft" : "ChevronRight"} 
              size={20} 
            />
            {isExpanded && (
              <span className="ml-2 text-sm">Recolher</span>
            )}
          </button>
        </div>
      </div>
      {/* Mobile Bottom Navigation */}
      <div className="fixed bottom-0 left-0 right-0 bg-card border-t border-border z-100 md:hidden">
        <div className="flex items-center justify-around py-2">
          {navigationItems?.map((item) => {
            const isActive = isActiveRoute(item?.path) || hasActiveSubItem(item);
            
            return (
              <button
                key={item?.path}
                onClick={() => handleNavigation(item?.path)}
                className={`flex flex-col items-center px-3 py-2 rounded-lg transition-smooth ${
                  isActive 
                    ? 'text-primary' :'text-muted-foreground'
                }`}
              >
                <Icon name={item?.icon} size={20} />
                <span className="text-xs mt-1 font-medium">
                  {item?.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </>
  );
};

export default NavigationSidebar;