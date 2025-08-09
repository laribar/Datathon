import React from 'react';
import Icon from '../../../components/AppIcon';


const RecommendationEngine = ({ recommendations, candidateStrengths, candidateWeaknesses }) => {
  const interviewFocusAreas = [
    {
      category: "Pontos Fortes",
      icon: "TrendingUp",
      color: "text-success",
      bgColor: "bg-success/10",
      items: candidateStrengths
    },
    {
      category: "Áreas de Atenção",
      icon: "AlertTriangle",
      color: "text-warning",
      bgColor: "bg-warning/10",
      items: candidateWeaknesses
    }
  ];

  const actionItems = [
    {
      title: "Perguntas Técnicas Sugeridas",
      icon: "MessageSquare",
      items: [
        "Como você implementaria autenticação JWT em uma aplicação React?",
        "Explique a diferença entre useState e useReducer.",
        "Descreva sua experiência com testes automatizados.",
        "Como você otimizaria a performance de uma aplicação web?"
      ]
    },
    {
      title: "Cenários Práticos",
      icon: "Code",
      items: [
        "Debugging de um componente React com performance lenta",
        "Implementação de um sistema de cache para APIs",
        "Resolução de conflitos em merge de código",
        "Arquitetura de uma aplicação escalável"
      ]
    },
    {
      title: "Avaliação Comportamental",
      icon: "Users",
      items: [
        "Experiência trabalhando em equipes ágeis",
        "Como lida com prazos apertados e pressão",
        "Exemplos de liderança técnica",
        "Adaptabilidade a novas tecnologias"
      ]
    }
  ];

  return (
    <div className="bg-card rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-accent/10 rounded-lg">
          <Icon name="Lightbulb" size={20} className="text-accent" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Engine de Recomendações</h3>
          <p className="text-sm text-muted-foreground">Insights IA para entrevista focada</p>
        </div>
      </div>
      {/* Overall Recommendation */}
      <div className="mb-6 p-4 bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg border border-primary/20">
        <div className="flex items-start gap-3">
          <Icon name="Target" size={20} className="text-primary mt-0.5" />
          <div>
            <h4 className="font-medium text-foreground mb-2">Recomendação Principal</h4>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {recommendations?.overall}
            </p>
          </div>
        </div>
      </div>
      {/* Strengths and Weaknesses */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {interviewFocusAreas?.map((area, index) => (
          <div key={index} className={`p-4 rounded-lg ${area?.bgColor} border border-current border-opacity-20`}>
            <div className="flex items-center gap-2 mb-3">
              <Icon name={area?.icon} size={16} className={area?.color} />
              <h4 className={`font-medium ${area?.color}`}>{area?.category}</h4>
            </div>
            <ul className="space-y-2">
              {area?.items?.map((item, itemIndex) => (
                <li key={itemIndex} className="flex items-start gap-2">
                  <Icon name="ArrowRight" size={12} className={`${area?.color} mt-1 flex-shrink-0`} />
                  <span className="text-sm text-foreground">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      {/* Interview Focus Areas */}
      <div className="space-y-4 mb-6">
        {actionItems?.map((section, index) => (
          <div key={index} className="border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <Icon name={section?.icon} size={16} className="text-primary" />
              <h4 className="font-medium text-foreground">{section?.title}</h4>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
              {section?.items?.map((item, itemIndex) => (
                <div
                  key={itemIndex}
                  className="p-3 bg-muted/30 rounded border border-border hover:border-primary/50 transition-colors"
                >
                  <span className="text-sm text-foreground">{item}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      {/* Next Steps */}
      <div className="p-4 bg-muted/30 rounded-lg">
        <h4 className="font-medium text-foreground mb-3 flex items-center gap-2">
          <Icon name="CheckSquare" size={16} className="text-primary" />
          Próximos Passos Recomendados
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-center gap-2 p-2 bg-card rounded border border-border">
            <Icon name="Calendar" size={14} className="text-success" />
            <span className="text-sm text-foreground">Agendar entrevista técnica</span>
          </div>
          <div className="flex items-center gap-2 p-2 bg-card rounded border border-border">
            <Icon name="FileText" size={14} className="text-primary" />
            <span className="text-sm text-foreground">Preparar teste prático</span>
          </div>
          <div className="flex items-center gap-2 p-2 bg-card rounded border border-border">
            <Icon name="Users" size={14} className="text-accent" />
            <span className="text-sm text-foreground">Envolver equipe técnica</span>
          </div>
          <div className="flex items-center gap-2 p-2 bg-card rounded border border-border">
            <Icon name="Clock" size={14} className="text-warning" />
            <span className="text-sm text-foreground">Definir cronograma</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationEngine;