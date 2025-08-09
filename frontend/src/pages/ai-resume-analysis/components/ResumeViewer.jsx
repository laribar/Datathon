import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const ResumeViewer = ({ resumeData, highlightedKeywords }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);

  const mockPdfPages = [
    "https://images.unsplash.com/photo-1586281380349-632531db7ed4?w=600&h=800&fit=crop",
    "https://images.unsplash.com/photo-1586281380614-a1d0e83cbf2c?w=600&h=800&fit=crop"
  ];

  const extractedInfo = [
    {
      category: "Informações Pessoais",
      icon: "User",
      items: [
        { label: "Nome", value: resumeData?.name, highlighted: true },
        { label: "Email", value: resumeData?.email, highlighted: false },
        { label: "Telefone", value: resumeData?.phone, highlighted: false },
        { label: "Localização", value: resumeData?.location, highlighted: false }
      ]
    },
    {
      category: "Habilidades Técnicas",
      icon: "Code",
      items: resumeData?.skills?.map(skill => ({
        label: skill,
        value: "Identificado",
        highlighted: highlightedKeywords?.includes(skill?.toLowerCase())
      }))
    },
    {
      category: "Experiência",
      icon: "Briefcase",
      items: resumeData?.experience?.map(exp => ({
        label: exp?.position,
        value: `${exp?.company} (${exp?.duration})`,
        highlighted: highlightedKeywords?.some(keyword => 
          exp?.position?.toLowerCase()?.includes(keyword) || 
          exp?.company?.toLowerCase()?.includes(keyword)
        )
      }))
    },
    {
      category: "Educação",
      icon: "GraduationCap",
      items: resumeData?.education?.map(edu => ({
        label: edu?.degree,
        value: `${edu?.institution} (${edu?.year})`,
        highlighted: highlightedKeywords?.some(keyword => 
          edu?.degree?.toLowerCase()?.includes(keyword) || 
          edu?.institution?.toLowerCase()?.includes(keyword)
        )
      }))
    }
  ];

  return (
    <div className="bg-card rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-destructive/10 rounded-lg">
            <Icon name="FileText" size={20} className="text-destructive" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Currículo PDF</h3>
            <p className="text-sm text-muted-foreground">Análise IA com extração de dados</p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          iconName={isExpanded ? "Minimize2" : "Maximize2"}
          iconPosition="left"
        >
          {isExpanded ? 'Minimizar' : 'Expandir'}
        </Button>
      </div>
      <div className={`transition-all duration-300 ${isExpanded ? 'h-96' : 'h-48'}`}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
          {/* PDF Preview */}
          <div className="bg-muted/30 rounded-lg p-4 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-foreground">
                Página {currentPage} de {mockPdfPages?.length}
              </span>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  iconName="ChevronLeft"
                />
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => setCurrentPage(Math.min(mockPdfPages?.length, currentPage + 1))}
                  disabled={currentPage === mockPdfPages?.length}
                  iconName="ChevronRight"
                />
              </div>
            </div>
            
            <div className="flex-1 bg-white rounded border overflow-hidden">
              <img
                src={mockPdfPages?.[currentPage - 1]}
                alt={`Página ${currentPage} do currículo`}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Extracted Information */}
          <div className="space-y-4 overflow-y-auto">
            {extractedInfo?.map((section, sectionIndex) => (
              <div key={sectionIndex} className="border border-border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Icon name={section?.icon} size={16} className="text-primary" />
                  <h4 className="font-medium text-foreground">{section?.category}</h4>
                </div>
                
                <div className="space-y-2">
                  {section?.items?.map((item, itemIndex) => (
                    <div
                      key={itemIndex}
                      className={`flex items-start justify-between p-2 rounded ${
                        item?.highlighted 
                          ? 'bg-success/10 border border-success/30' :'bg-muted/30'
                      }`}
                    >
                      <div className="flex-1">
                        <span className="text-sm font-medium text-foreground">
                          {item?.label}
                        </span>
                        {item?.value && (
                          <p className="text-xs text-muted-foreground mt-1">
                            {item?.value}
                          </p>
                        )}
                      </div>
                      {item?.highlighted && (
                        <Icon name="Check" size={14} className="text-success mt-0.5" />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      {/* Action Buttons */}
      <div className="flex items-center gap-3 mt-6 pt-6 border-t border-border">
        <Button
          variant="outline"
          size="sm"
          iconName="Download"
          iconPosition="left"
        >
          Baixar PDF
        </Button>
        <Button
          variant="outline"
          size="sm"
          iconName="Eye"
          iconPosition="left"
        >
          Visualizar Completo
        </Button>
        <Button
          variant="outline"
          size="sm"
          iconName="Share"
          iconPosition="left"
        >
          Compartilhar
        </Button>
      </div>
    </div>
  );
};

export default ResumeViewer;