import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const InterviewHeader = ({ 
  candidateName = "Carlos Santos",
  position = "Desenvolvedor React Sênior",
  interviewType = "Técnica",
  startTime = new Date()
}) => {
  const navigate = useNavigate();
  const [showCandidateInfo, setShowCandidateInfo] = useState(false);

  const candidateInfo = {
    name: "Carlos Santos",
    email: "carlos.santos@email.com",
    phone: "+55 11 99999-9999",
    experience: "5 anos",
    location: "São Paulo, SP",
    linkedin: "linkedin.com/in/carlos-santos",
    skills: ["React", "JavaScript", "TypeScript", "Node.js", "MongoDB"],
    education: "Bacharelado em Ciência da Computação - USP",
    currentRole: "Desenvolvedor Full Stack na TechCorp"
  };

  const formatTime = (date) => {
    return date?.toLocaleTimeString('pt-BR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <>
      <header className="bg-card border-b border-border px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left Section - Navigation */}
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate('/candidate-management')}
              className="h-10 w-10"
            >
              <Icon name="ArrowLeft" size={20} />
            </Button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <Icon name="Video" size={20} className="text-white" />
              </div>
              <div>
                <h1 className="text-lg font-semibold text-foreground">Sala de Entrevista</h1>
                <p className="text-sm text-muted-foreground">
                  Entrevista {interviewType} • Iniciada às {formatTime(startTime)}
                </p>
              </div>
            </div>
          </div>

          {/* Center Section - Candidate Info */}
          <div className="flex items-center gap-4">
            <div 
              className="flex items-center gap-3 px-4 py-2 bg-muted/50 rounded-lg cursor-pointer hover:bg-muted/70 transition-colors"
              onClick={() => setShowCandidateInfo(true)}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-semibold">
                  {candidateName?.split(' ')?.map(n => n?.[0])?.join('')}
                </span>
              </div>
              <div>
                <p className="font-medium text-foreground">{candidateName}</p>
                <p className="text-sm text-muted-foreground">{position}</p>
              </div>
              <Icon name="ChevronDown" size={16} className="text-muted-foreground" />
            </div>
          </div>

          {/* Right Section - Actions */}
          <div className="flex items-center gap-3">
            <Button variant="outline" size="sm">
              <Icon name="FileText" size={16} className="mr-2" />
              Currículo
            </Button>
            
            <Button variant="outline" size="sm">
              <Icon name="Users" size={16} className="mr-2" />
              Convidar
            </Button>

            <div className="w-px h-6 bg-border"></div>

            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10"
            >
              <Icon name="MoreVertical" size={20} />
            </Button>
          </div>
        </div>
      </header>
      {/* Candidate Info Modal */}
      {showCandidateInfo && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-card rounded-lg border border-border w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-border">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-blue-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-xl font-semibold">
                    {candidateInfo?.name?.split(' ')?.map(n => n?.[0])?.join('')}
                  </span>
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-foreground">{candidateInfo?.name}</h2>
                  <p className="text-muted-foreground">{position}</p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowCandidateInfo(false)}
              >
                <Icon name="X" size={20} />
              </Button>
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-6">
              {/* Contact Information */}
              <div>
                <h3 className="font-semibold text-foreground mb-3">Informações de Contato</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center gap-3">
                    <Icon name="Mail" size={16} className="text-muted-foreground" />
                    <span className="text-sm text-foreground">{candidateInfo?.email}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Icon name="Phone" size={16} className="text-muted-foreground" />
                    <span className="text-sm text-foreground">{candidateInfo?.phone}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Icon name="MapPin" size={16} className="text-muted-foreground" />
                    <span className="text-sm text-foreground">{candidateInfo?.location}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Icon name="Linkedin" size={16} className="text-muted-foreground" />
                    <span className="text-sm text-foreground">{candidateInfo?.linkedin}</span>
                  </div>
                </div>
              </div>

              {/* Professional Information */}
              <div>
                <h3 className="font-semibold text-foreground mb-3">Informações Profissionais</h3>
                <div className="space-y-3">
                  <div>
                    <span className="text-sm font-medium text-muted-foreground">Experiência:</span>
                    <span className="text-sm text-foreground ml-2">{candidateInfo?.experience}</span>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-muted-foreground">Cargo Atual:</span>
                    <span className="text-sm text-foreground ml-2">{candidateInfo?.currentRole}</span>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-muted-foreground">Educação:</span>
                    <span className="text-sm text-foreground ml-2">{candidateInfo?.education}</span>
                  </div>
                </div>
              </div>

              {/* Skills */}
              <div>
                <h3 className="font-semibold text-foreground mb-3">Habilidades Técnicas</h3>
                <div className="flex flex-wrap gap-2">
                  {candidateInfo?.skills?.map((skill, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 rounded-full text-sm"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              {/* Quick Actions */}
              <div className="flex gap-3 pt-4 border-t border-border">
                <Button variant="outline" className="flex-1">
                  <Icon name="FileText" size={16} className="mr-2" />
                  Ver Currículo
                </Button>
                <Button variant="outline" className="flex-1">
                  <Icon name="Star" size={16} className="mr-2" />
                  Avaliar
                </Button>
                <Button variant="outline" className="flex-1">
                  <Icon name="MessageSquare" size={16} className="mr-2" />
                  Notas
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default InterviewHeader;