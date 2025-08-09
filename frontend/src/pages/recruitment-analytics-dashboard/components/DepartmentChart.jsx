import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const DepartmentChart = () => {
  const departmentData = [
    { name: 'Engenharia', value: 45, color: '#2F9FF8', hires: 28 },
    { name: 'Marketing', value: 20, color: '#2ECC71', hires: 12 },
    { name: 'Vendas', value: 18, color: '#8B5CF6', hires: 11 },
    { name: 'RH', value: 8, color: '#F59E0B', hires: 5 },
    { name: 'Financeiro', value: 6, color: '#E74C3C', hires: 4 },
    { name: 'Outros', value: 3, color: '#64748B', hires: 2 }
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload?.length) {
      const data = payload?.[0]?.payload;
      return (
        <div className="bg-popover border border-border rounded-lg p-3 shadow-elevation-2">
          <p className="text-foreground font-medium">{data?.name}</p>
          <p className="text-primary font-bold">{data?.value}% das contratações</p>
          <p className="text-muted-foreground text-sm">{data?.hires} contratações</p>
        </div>
      );
    }
    return null;
  };

  const CustomLegend = ({ payload }) => {
    return (
      <div className="flex flex-wrap justify-center gap-4 mt-4">
        {payload?.map((entry, index) => (
          <div key={index} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: entry?.color }}
            />
            <span className="text-sm text-muted-foreground">
              {entry?.value} ({departmentData?.find(d => d?.name === entry?.value)?.value}%)
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground">Contratações por Departamento</h3>
        <div className="text-sm text-muted-foreground">
          Total: 62 contratações
        </div>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={departmentData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={120}
              paddingAngle={2}
              dataKey="value"
              animationBegin={0}
              animationDuration={1000}
            >
              {departmentData?.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry?.color} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend content={<CustomLegend />} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6 pt-6 border-t border-border">
        {departmentData?.map((dept) => (
          <div key={dept?.name} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg">
            <div className="flex items-center gap-3">
              <div 
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: dept?.color }}
              />
              <span className="text-sm text-foreground">{dept?.name}</span>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-foreground">{dept?.hires}</p>
              <p className="text-xs text-muted-foreground">{dept?.value}%</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DepartmentChart;