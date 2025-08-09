import React from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const Header = () => {
  return (
    <header className="bg-card border-b border-border px-6 py-4 sticky top-0 z-40">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-lg font-semibold text-foreground">Criação de Vaga</h1>
            <p className="text-sm text-muted-foreground">
              Configure todos os detalhes da nova oportunidade
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Notifications */}
          <Button variant="ghost" size="sm" className="relative">
            <Icon name="Bell" size={20} />
            <span className="absolute -top-1 -right-1 w-2 h-2 bg-destructive rounded-full"></span>
          </Button>

          {/* Help */}
          <Button variant="ghost" size="sm">
            <Icon name="HelpCircle" size={20} />
          </Button>

          {/* Quick Stats */}
          <div className="hidden md:flex items-center gap-6 ml-6 pl-6 border-l border-border">
            <div className="text-center">
              <div className="text-lg font-semibold text-foreground">12</div>
              <div className="text-xs text-muted-foreground">Vagas Ativas</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-success">248</div>
              <div className="text-xs text-muted-foreground">Candidatos</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-primary">89%</div>
              <div className="text-xs text-muted-foreground">Match Rate</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;