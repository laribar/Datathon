import React from 'react';
import Icon from '../../../components/AppIcon';

const KeyTopicsExtraction = ({ topicsData }) => {
  const getTopicColor = (relevance) => {
    if (relevance >= 80) return 'bg-success/20 text-success border-success/30';
    if (relevance >= 60) return 'bg-warning/20 text-warning border-warning/30';
    return 'bg-muted/20 text-muted-foreground border-muted/30';
  };

  return (
    <div className="bg-card rounded-lg border border-border p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-foreground">Tópicos Principais</h3>
        <Icon name="Hash" size={20} className="text-muted-foreground" />
      </div>
      <div className="space-y-4">
        {topicsData?.map((topic, index) => (
          <div key={index} className="border border-border rounded-lg p-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <h4 className="font-medium text-foreground mb-1">{topic?.title}</h4>
                <p className="text-sm text-muted-foreground">{topic?.description}</p>
              </div>
              <div className="ml-4 text-right">
                <div className="text-sm text-muted-foreground mb-1">Relevância</div>
                <div className="text-lg font-semibold text-primary">{topic?.relevance}%</div>
              </div>
            </div>
            
            <div className="flex flex-wrap gap-2 mb-3">
              {topic?.keywords?.map((keyword, keyIndex) => (
                <span 
                  key={keyIndex}
                  className={`px-2 py-1 rounded-full text-xs border ${getTopicColor(topic?.relevance)}`}
                >
                  {keyword}
                </span>
              ))}
            </div>
            
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-4">
                <span className="text-muted-foreground">
                  <Icon name="Clock" size={14} className="inline mr-1" />
                  {topic?.duration}min
                </span>
                <span className="text-muted-foreground">
                  <Icon name="MessageSquare" size={14} className="inline mr-1" />
                  {topic?.mentions} menções
                </span>
              </div>
              <button className="text-primary hover:text-primary/80 transition-colors">
                Ver detalhes
              </button>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-6 p-4 bg-muted/30 rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <Icon name="BarChart3" size={18} className="text-primary" />
          <span className="font-medium text-foreground">Resumo da Análise</span>
        </div>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-lg font-semibold text-primary">
              {topicsData?.length}
            </div>
            <div className="text-xs text-muted-foreground">Tópicos</div>
          </div>
          <div>
            <div className="text-lg font-semibold text-success">
              {topicsData?.filter(t => t?.relevance >= 80)?.length}
            </div>
            <div className="text-xs text-muted-foreground">Alta Relevância</div>
          </div>
          <div>
            <div className="text-lg font-semibold text-warning">
              {topicsData?.reduce((acc, t) => acc + t?.mentions, 0)}
            </div>
            <div className="text-xs text-muted-foreground">Total Menções</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default KeyTopicsExtraction;