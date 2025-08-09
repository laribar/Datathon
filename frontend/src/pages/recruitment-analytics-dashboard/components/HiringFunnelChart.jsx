import React from 'react';
import { FunnelChart, Funnel, Cell, ResponsiveContainer, Tooltip, LabelList } from 'recharts';

const HiringFunnelChart = () => {
  const funnelData = [
    { name: 'Candidatos Aplicados', value: 1250, color: '#2F9FF8' },
    { name: 'Triagem Inicial', value: 850, color: '#3B82F6' },
    { name: 'Entrevista Técnica', value: 420, color: '#6366F1' },
    { name: 'Entrevista Final', value: 180, color: '#8B5CF6' },
    { name: 'Ofertas Enviadas', value: 95, color: '#A855F7' },
    { name: 'Contratações', value: 72, color: '#2ECC71' }
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload?.length) {
      const data = payload?.[0]?.payload;
      const prevValue = funnelData?.[funnelData?.findIndex(item => item?.name === data?.name) - 1]?.value;
      const conversionRate = prevValue ? ((data?.value / prevValue) * 100)?.toFixed(1) : 100;
      
      return (
        <div className="bg-popover border border-border rounded-lg p-3 shadow-elevation-2">
          <p className="text-foreground font-medium">{data?.name}</p>
          <p className="text-primary font-bold">{data?.value?.toLocaleString('pt-BR')} candidatos</p>
          <p className="text-muted-foreground text-sm">Taxa de conversão: {conversionRate}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground">Funil de Contratação</h3>
        <div className="text-sm text-muted-foreground">
          Taxa de conversão geral: 5.8%
        </div>
      </div>
      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <FunnelChart>
            <Tooltip content={<CustomTooltip />} />
            <Funnel
              dataKey="value"
              data={funnelData}
              isAnimationActive={true}
              animationDuration={1000}
            >
              {funnelData?.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry?.color} />
              ))}
              <LabelList 
                position="center" 
                fill="#fff" 
                stroke="none" 
                fontSize={12}
                formatter={(value, entry) => `${entry?.name}\n${value?.toLocaleString('pt-BR')}`}
              />
            </Funnel>
          </FunnelChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6 pt-6 border-t border-border">
        {funnelData?.map((item, index) => {
          const prevValue = index > 0 ? funnelData?.[index - 1]?.value : item?.value;
          const conversionRate = ((item?.value / prevValue) * 100)?.toFixed(1);
          
          return (
            <div key={item?.name} className="text-center">
              <div 
                className="w-4 h-4 rounded-full mx-auto mb-2"
                style={{ backgroundColor: item?.color }}
              />
              <p className="text-xs text-muted-foreground mb-1">{item?.name}</p>
              <p className="text-sm font-medium text-foreground">{item?.value?.toLocaleString('pt-BR')}</p>
              {index > 0 && (
                <p className="text-xs text-muted-foreground">{conversionRate}%</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default HiringFunnelChart;