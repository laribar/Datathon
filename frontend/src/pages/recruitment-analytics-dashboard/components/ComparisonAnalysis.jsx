import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const ComparisonAnalysis = () => {
  const [viewType, setViewType] = useState('yoy'); // yoy = year-over-year, benchmark = industry

  const yoyData = [
    { metric: 'Tempo de Contratação', atual: 28, anterior: 35, industry: 32 },
    { metric: 'Custo por Contratação', atual: 8500, anterior: 9200, industry: 9800 },
    { metric: 'Taxa de Conversão', atual: 5.8, anterior: 4.2, industry: 4.9 },
    { metric: 'Satisfação do Candidato', atual: 4.6, anterior: 4.1, industry: 4.3 },
    { metric: 'Retenção 90 dias', atual: 92, anterior: 88, industry: 85 }
  ];

  const benchmarkData = [
    { category: 'Tecnologia', nossa: 28, mercado: 32, diferenca: -4 },
    { category: 'Startups', nossa: 28, mercado: 25, diferenca: 3 },
    { category: 'Empresas Médias', nossa: 28, mercado: 35, diferenca: -7 },
    { category: 'Multinacionais', nossa: 28, mercado: 42, diferenca: -14 }
  ];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload?.length) {
      return (
        <div className="bg-popover border border-border rounded-lg p-3 shadow-elevation-2">
          <p className="text-foreground font-medium mb-2">{label}</p>
          {payload?.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry?.color }}>
              {entry?.name}: {entry?.value?.toLocaleString('pt-BR')}
              {entry?.dataKey === 'atual'|| entry?.dataKey === 'anterior' ? (label?.includes('Custo') ? ' R$' : 
                 label?.includes('Taxa') || label?.includes('Satisfação') || label?.includes('Retenção') ? '%' : 
                 ' dias') : ''}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const getPerformanceIndicator = (current, previous) => {
    const isImprovement = current > previous;
    const isCostMetric = false; // Simplified for this example
    const isGood = isCostMetric ? current < previous : current > previous;
    
    return {
      icon: isGood ? 'TrendingUp' : 'TrendingDown',
      color: isGood ? 'text-green-400' : 'text-red-400',
      change: Math.abs(((current - previous) / previous * 100))?.toFixed(1)
    };
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <Icon name="BarChart3" size={20} />
          Análise Comparativa
        </h3>
        <div className="flex gap-2">
          <Button
            variant={viewType === 'yoy' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewType('yoy')}
          >
            Ano a Ano
          </Button>
          <Button
            variant={viewType === 'benchmark' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewType('benchmark')}
          >
            Benchmark
          </Button>
        </div>
      </div>
      {viewType === 'yoy' && (
        <>
          <div className="h-64 mb-6">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={yoyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis 
                  dataKey="metric" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#94A3B8', fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#94A3B8', fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="anterior" 
                  name="Ano Anterior"
                  fill="#64748B" 
                  radius={[4, 4, 0, 0]}
                  animationDuration={1000}
                />
                <Bar 
                  dataKey="atual" 
                  name="Ano Atual"
                  fill="#2F9FF8" 
                  radius={[4, 4, 0, 0]}
                  animationDuration={1200}
                />
                <Bar 
                  dataKey="industry" 
                  name="Média do Mercado"
                  fill="#8B5CF6" 
                  radius={[4, 4, 0, 0]}
                  animationDuration={1400}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {yoyData?.slice(0, 3)?.map((item, index) => {
              const indicator = getPerformanceIndicator(item?.atual, item?.anterior);
              return (
                <div key={index} className="bg-muted/20 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">{item?.metric}</span>
                    <div className={`flex items-center gap-1 ${indicator?.color}`}>
                      <Icon name={indicator?.icon} size={14} />
                      <span className="text-xs">{indicator?.change}%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-lg font-bold text-foreground">
                        {item?.atual?.toLocaleString('pt-BR')}
                        {item?.metric?.includes('Custo') ? ' R$' : 
                         item?.metric?.includes('Taxa') || item?.metric?.includes('Satisfação') || item?.metric?.includes('Retenção') ? '%' : 
                         ' dias'}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        vs {item?.anterior?.toLocaleString('pt-BR')} anterior
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
      {viewType === 'benchmark' && (
        <>
          <div className="space-y-4 mb-6">
            <div className="text-center">
              <h4 className="text-sm font-medium text-foreground mb-2">
                Tempo de Contratação - Comparação por Segmento
              </h4>
              <p className="text-xs text-muted-foreground">
                Nossa performance vs média do mercado (em dias)
              </p>
            </div>
            
            {benchmarkData?.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-muted/20 rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-foreground">{item?.category}</span>
                    <div className={`flex items-center gap-1 text-xs ${
                      item?.diferenca < 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      <Icon name={item?.diferenca < 0 ? 'TrendingUp' : 'TrendingDown'} size={12} />
                      {Math.abs(item?.diferenca)} dias {item?.diferenca < 0 ? 'melhor' : 'pior'}
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Nossa empresa</span>
                        <span className="text-foreground font-medium">{item?.nossa} dias</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${(item?.nossa / Math.max(item?.nossa, item?.mercado)) * 100}%` }}
                        />
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-muted-foreground">Mercado</span>
                        <span className="text-foreground font-medium">{item?.mercado} dias</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2">
                        <div 
                          className="bg-purple-500 h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${(item?.mercado / Math.max(item?.nossa, item?.mercado)) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Icon name="Award" size={16} className="text-green-400" />
              <span className="text-sm font-medium text-green-400">Performance Destacada</span>
            </div>
            <p className="text-sm text-foreground">
              Nossa empresa está 23% mais rápida que a média do mercado de tecnologia, 
              demonstrando eficiência superior nos processos de recrutamento.
            </p>
          </div>
        </>
      )}
    </div>
  );
};

export default ComparisonAnalysis;