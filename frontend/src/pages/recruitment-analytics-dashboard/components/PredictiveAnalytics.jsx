import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Icon from '../../../components/AppIcon';

const PredictiveAnalytics = () => {
  const forecastData = [
    { month: 'Jan', historical: 12, predicted: null, confidence: null },
    { month: 'Fev', historical: 15, predicted: null, confidence: null },
    { month: 'Mar', historical: 18, predicted: null, confidence: null },
    { month: 'Abr', historical: 22, predicted: null, confidence: null },
    { month: 'Mai', historical: 19, predicted: null, confidence: null },
    { month: 'Jun', historical: 25, predicted: null, confidence: null },
    { month: 'Jul', historical: 28, predicted: null, confidence: null },
    { month: 'Ago', historical: 32, predicted: null, confidence: null },
    { month: 'Set', historical: null, predicted: 35, confidence: { min: 30, max: 40 } },
    { month: 'Out', historical: null, predicted: 38, confidence: { min: 32, max: 44 } },
    { month: 'Nov', historical: null, predicted: 42, confidence: { min: 35, max: 49 } },
    { month: 'Dez', historical: null, predicted: 45, confidence: { min: 38, max: 52 } }
  ];

  const insights = [
    {
      icon: 'TrendingUp',
      title: 'Crescimento Previsto',
      value: '+18%',
      description: 'Aumento esperado nas contratações para Q4',
      color: 'text-green-400'
    },
    {
      icon: 'Clock',
      title: 'Tempo Médio de Contratação',
      value: '28 dias',
      description: 'Redução de 3 dias comparado ao trimestre anterior',
      color: 'text-blue-400'
    },
    {
      icon: 'Users',
      title: 'Demanda por Talentos',
      value: 'Alta',
      description: 'Especialmente em Engenharia e Marketing',
      color: 'text-purple-400'
    },
    {
      icon: 'Target',
      title: 'Meta Q4',
      value: '160',
      description: 'Contratações planejadas para o próximo trimestre',
      color: 'text-orange-400'
    }
  ];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload?.length) {
      const data = forecastData?.find(d => d?.month === label);
      return (
        <div className="bg-popover border border-border rounded-lg p-3 shadow-elevation-2">
          <p className="text-foreground font-medium mb-2">{label} 2024</p>
          {data?.historical && (
            <p className="text-blue-400 text-sm">
              Histórico: {data?.historical} contratações
            </p>
          )}
          {data?.predicted && (
            <>
              <p className="text-green-400 text-sm">
                Previsão: {data?.predicted} contratações
              </p>
              {data?.confidence && (
                <p className="text-muted-foreground text-xs">
                  Intervalo: {data?.confidence?.min}-{data?.confidence?.max}
                </p>
              )}
            </>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <Icon name="Brain" size={20} />
          Análise Preditiva
        </h3>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Icon name="Info" size={16} />
          Baseado em dados históricos e tendências
        </div>
      </div>
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={forecastData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis 
              dataKey="month" 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94A3B8', fontSize: 12 }}
            />
            <YAxis 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94A3B8', fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="historical"
              stroke="#2F9FF8"
              strokeWidth={3}
              dot={{ fill: '#2F9FF8', strokeWidth: 2, r: 4 }}
              connectNulls={false}
              animationDuration={1000}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#2ECC71"
              strokeWidth={3}
              strokeDasharray="5 5"
              dot={{ fill: '#2ECC71', strokeWidth: 2, r: 4 }}
              connectNulls={false}
              animationDuration={1200}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {insights?.map((insight, index) => (
          <div key={index} className="bg-muted/20 rounded-lg p-4">
            <div className="flex items-center gap-3 mb-2">
              <Icon name={insight?.icon} size={20} className={insight?.color} />
              <span className="text-sm font-medium text-foreground">{insight?.title}</span>
            </div>
            <p className={`text-xl font-bold mb-1 ${insight?.color}`}>{insight?.value}</p>
            <p className="text-xs text-muted-foreground">{insight?.description}</p>
          </div>
        ))}
      </div>
      <div className="mt-6 pt-6 border-t border-border">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-foreground mb-1">Recomendações da IA</h4>
            <p className="text-xs text-muted-foreground">
              Baseado na análise de padrões e tendências históricas
            </p>
          </div>
          <Icon name="Sparkles" size={20} className="text-yellow-400" />
        </div>
        <div className="mt-4 space-y-2">
          <div className="flex items-start gap-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <Icon name="Lightbulb" size={16} className="text-blue-400 mt-0.5" />
            <div>
              <p className="text-sm text-foreground">
                Considere aumentar o orçamento de recrutamento em 15% para Q4
              </p>
              <p className="text-xs text-muted-foreground">
                Para atender à demanda prevista de contratações
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
            <Icon name="Users" size={16} className="text-green-400 mt-0.5" />
            <div>
              <p className="text-sm text-foreground">
                Foque em sourcing proativo para posições de Engenharia
              </p>
              <p className="text-xs text-muted-foreground">
                Maior demanda esperada neste departamento
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictiveAnalytics;