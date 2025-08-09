import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const MonthlyVolumeChart = () => {
  const monthlyData = [
    { month: 'Jan', aplicacoes: 180, contratacoes: 12, ofertas: 18 },
    { month: 'Fev', aplicacoes: 220, contratacoes: 15, ofertas: 22 },
    { month: 'Mar', aplicacoes: 280, contratacoes: 18, ofertas: 28 },
    { month: 'Abr', aplicacoes: 320, contratacoes: 22, ofertas: 32 },
    { month: 'Mai', aplicacoes: 290, contratacoes: 19, ofertas: 29 },
    { month: 'Jun', aplicacoes: 350, contratacoes: 25, ofertas: 38 },
    { month: 'Jul', aplicacoes: 380, contratacoes: 28, ofertas: 42 },
    { month: 'Ago', aplicacoes: 420, contratacoes: 32, ofertas: 48 },
    { month: 'Set', aplicacoes: 390, contratacoes: 29, ofertas: 44 },
    { month: 'Out', aplicacoes: 450, contratacoes: 35, ofertas: 52 },
    { month: 'Nov', aplicacoes: 480, contratacoes: 38, ofertas: 56 },
    { month: 'Dez', aplicacoes: 520, contratacoes: 42, ofertas: 62 }
  ];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload?.length) {
      return (
        <div className="bg-popover border border-border rounded-lg p-3 shadow-elevation-2">
          <p className="text-foreground font-medium mb-2">{label} 2024</p>
          {payload?.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry?.color }}>
              {entry?.name}: {entry?.value?.toLocaleString('pt-BR')}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground">Volume Mensal de Recrutamento</h3>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full" />
            <span className="text-muted-foreground">Aplicações</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full" />
            <span className="text-muted-foreground">Contratações</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-purple-500 rounded-full" />
            <span className="text-muted-foreground">Ofertas</span>
          </div>
        </div>
      </div>
      
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={monthlyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
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
            <Bar 
              dataKey="aplicacoes" 
              name="Aplicações"
              fill="#2F9FF8" 
              radius={[4, 4, 0, 0]}
              animationDuration={1000}
            />
            <Bar 
              dataKey="ofertas" 
              name="Ofertas"
              fill="#8B5CF6" 
              radius={[4, 4, 0, 0]}
              animationDuration={1200}
            />
            <Bar 
              dataKey="contratacoes" 
              name="Contratações"
              fill="#2ECC71" 
              radius={[4, 4, 0, 0]}
              animationDuration={1400}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-border">
        <div className="text-center">
          <p className="text-2xl font-bold text-blue-400">4.290</p>
          <p className="text-sm text-muted-foreground">Total de Aplicações</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-purple-400">471</p>
          <p className="text-sm text-muted-foreground">Total de Ofertas</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-green-400">315</p>
          <p className="text-sm text-muted-foreground">Total de Contratações</p>
        </div>
      </div>
    </div>
  );
};

export default MonthlyVolumeChart;