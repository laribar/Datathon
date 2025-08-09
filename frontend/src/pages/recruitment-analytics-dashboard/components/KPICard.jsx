import React from 'react';
import Icon from '../../../components/AppIcon';

const KPICard = ({ title, value, change, changeType, icon, color = "blue" }) => {
  const getColorClasses = (color) => {
    const colors = {
      blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
      green: "bg-green-500/10 text-green-400 border-green-500/20",
      purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
      orange: "bg-orange-500/10 text-orange-400 border-orange-500/20"
    };
    return colors?.[color] || colors?.blue;
  };

  const getTrendIcon = (type) => {
    if (type === 'up') return 'TrendingUp';
    if (type === 'down') return 'TrendingDown';
    return 'Minus';
  };

  const getTrendColor = (type) => {
    if (type === 'up') return 'text-green-400';
    if (type === 'down') return 'text-red-400';
    return 'text-slate-400';
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6 hover:shadow-elevation-1 transition-smooth">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-lg border ${getColorClasses(color)}`}>
          <Icon name={icon} size={24} />
        </div>
        <div className={`flex items-center gap-1 text-sm ${getTrendColor(changeType)}`}>
          <Icon name={getTrendIcon(changeType)} size={16} />
          <span>{change}</span>
        </div>
      </div>
      <div>
        <h3 className="text-2xl font-bold text-foreground mb-1">{value}</h3>
        <p className="text-muted-foreground text-sm">{title}</p>
      </div>
    </div>
  );
};

export default KPICard;