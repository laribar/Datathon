import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Icon from '../AppIcon';

const WorkflowBreadcrumb = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const breadcrumbMappings = {
    '/candidate-management': [
      { label: 'Candidatos', path: '/candidate-management', icon: 'Users' }
    ],
    '/ai-resume-analysis': [
      { label: 'Candidatos', path: '/candidate-management', icon: 'Users' },
      { label: 'Análise de CV', path: '/ai-resume-analysis', icon: 'FileText' }
    ],
    '/video-interview-room': [
      { label: 'Entrevistas', path: '/video-interview-room', icon: 'Video' }
    ],
    '/interview-transcription-analysis': [
      { label: 'Entrevistas', path: '/video-interview-room', icon: 'Video' },
      { label: 'Análise de Transcrição', path: '/interview-transcription-analysis', icon: 'MessageSquare' }
    ]
  };

  const currentBreadcrumb = breadcrumbMappings?.[location?.pathname];

  if (!currentBreadcrumb || currentBreadcrumb?.length <= 1) {
    return null;
  }

  const handleNavigation = (path) => {
    navigate(path);
  };

  return (
    <nav className="flex items-center space-x-2 px-6 py-4 bg-background border-b border-border">
      <div className="flex items-center space-x-2 text-sm">
        {currentBreadcrumb?.map((item, index) => (
          <React.Fragment key={item?.path}>
            {index > 0 && (
              <Icon 
                name="ChevronRight" 
                size={16} 
                className="text-muted-foreground" 
              />
            )}
            <button
              onClick={() => handleNavigation(item?.path)}
              className={`flex items-center space-x-1 px-2 py-1 rounded-md transition-smooth ${
                index === currentBreadcrumb?.length - 1
                  ? 'text-foreground bg-muted cursor-default'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted'
              }`}
              disabled={index === currentBreadcrumb?.length - 1}
            >
              <Icon name={item?.icon} size={14} />
              <span className="font-medium">{item?.label}</span>
            </button>
          </React.Fragment>
        ))}
      </div>
    </nav>
  );
};

export default WorkflowBreadcrumb;