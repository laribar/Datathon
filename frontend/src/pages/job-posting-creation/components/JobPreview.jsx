import React from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const JobPreview = ({ formData }) => {
  const formatSalary = (min, max, negotiable) => {
    if (negotiable) return 'A combinar';
    if (min && max) return `R$ ${parseInt(min)?.toLocaleString('pt-BR')} - R$ ${parseInt(max)?.toLocaleString('pt-BR')}`;
    if (min) return `A partir de R$ ${parseInt(min)?.toLocaleString('pt-BR')}`;
    if (max) return `Até R$ ${parseInt(max)?.toLocaleString('pt-BR')}`;
    return 'Não informado';
  };

  const formatSkills = (skillsString) => {
    if (!skillsString) return [];
    return skillsString?.split(',')?.map(skill => skill?.trim())?.filter(skill => skill);
  };

  const getEmploymentTypeLabel = (type) => {
    const types = {
      'clt': 'CLT',
      'pj': 'PJ',
      'estagio': 'Estágio',
      'freelancer': 'Freelancer',
      'temporario': 'Temporário'
    };
    return types?.[type] || type;
  };

  const getLocationLabel = (location) => {
    const locations = {
      'sao-paulo': 'São Paulo, SP',
      'rio-janeiro': 'Rio de Janeiro, RJ',
      'belo-horizonte': 'Belo Horizonte, MG',
      'brasilia': 'Brasília, DF',
      'remoto': 'Remoto',
      'hibrido': 'Híbrido'
    };
    return locations?.[location] || location;
  };

  const getDepartmentLabel = (dept) => {
    const departments = {
      'tecnologia': 'Tecnologia',
      'marketing': 'Marketing',
      'vendas': 'Vendas',
      'rh': 'Recursos Humanos',
      'financeiro': 'Financeiro',
      'operacoes': 'Operações'
    };
    return departments?.[dept] || dept;
  };

  return (
    <div className="bg-card rounded-lg shadow-elevation-1 p-6 sticky top-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-foreground">Pré-visualização</h2>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Icon name="Eye" size={16} />
          <span>Visão do candidato</span>
        </div>
      </div>
      <div className="space-y-6">
        {/* Job Header */}
        <div className="space-y-3">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h1 className="text-xl font-bold text-foreground mb-2">
                {formData?.title || 'Título da Vaga'}
              </h1>
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Icon name="Building2" size={16} />
                  <span>{formData?.department ? getDepartmentLabel(formData?.department) : 'Departamento'}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Icon name="MapPin" size={16} />
                  <span>{formData?.location ? getLocationLabel(formData?.location) : 'Localização'}</span>
                </div>
              </div>
            </div>
            {formData?.urgent && (
              <div className="bg-warning/10 text-warning px-2 py-1 rounded-full text-xs font-medium">
                Urgente
              </div>
            )}
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1 text-muted-foreground">
              <Icon name="Briefcase" size={16} />
              <span>{formData?.employmentType ? getEmploymentTypeLabel(formData?.employmentType) : 'Tipo de contrato'}</span>
            </div>
            <div className="flex items-center gap-1 text-success font-medium">
              <Icon name="DollarSign" size={16} />
              <span>{formatSalary(formData?.salaryMin, formData?.salaryMax, formData?.salaryNegotiable)}</span>
            </div>
          </div>
        </div>

        {/* Job Description */}
        <div className="space-y-3">
          <h3 className="font-semibold text-foreground">Descrição da Vaga</h3>
          <div 
            className="prose prose-sm max-w-none text-muted-foreground"
            dangerouslySetInnerHTML={{ 
              __html: formData?.description || '<p className="text-muted-foreground italic">A descrição da vaga aparecerá aqui conforme você digita...</p>' 
            }}
          />
        </div>

        {/* Technical Skills */}
        {formData?.technicalSkills && (
          <div className="space-y-3">
            <h3 className="font-semibold text-foreground">Habilidades Técnicas</h3>
            <div className="flex flex-wrap gap-2">
              {formatSkills(formData?.technicalSkills)?.map((skill, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-primary/10 text-primary text-sm rounded-full"
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Soft Skills */}
        {formData?.softSkills && (
          <div className="space-y-3">
            <h3 className="font-semibold text-foreground">Habilidades Comportamentais</h3>
            <div className="flex flex-wrap gap-2">
              {formatSkills(formData?.softSkills)?.map((skill, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-accent/10 text-accent text-sm rounded-full"
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Education */}
        {formData?.education && (
          <div className="space-y-3">
            <h3 className="font-semibold text-foreground">Formação Acadêmica</h3>
            <p className="text-muted-foreground text-sm">{formData?.education}</p>
          </div>
        )}

        {/* Benefits */}
        {formData?.benefits && (
          <div className="space-y-3">
            <h3 className="font-semibold text-foreground">Benefícios</h3>
            <div className="flex flex-wrap gap-2">
              {formatSkills(formData?.benefits)?.map((benefit, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-success/10 text-success text-sm rounded-full"
                >
                  {benefit}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Experience Level */}
        {formData?.experience && (
          <div className="space-y-3">
            <h3 className="font-semibold text-foreground">Nível de Experiência</h3>
            <div className="flex items-center gap-2">
              <Icon name="TrendingUp" size={16} className="text-primary" />
              <span className="text-muted-foreground text-sm">
                {formData?.experience === 'junior' && 'Júnior (0-2 anos)'}
                {formData?.experience === 'pleno' && 'Pleno (2-5 anos)'}
                {formData?.experience === 'senior' && 'Sênior (5+ anos)'}
                {formData?.experience === 'especialista' && 'Especialista (8+ anos)'}
              </span>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="pt-4 border-t border-border space-y-3">
          <Button variant="default" fullWidth iconName="Send" iconPosition="left">
            Candidatar-se
          </Button>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" iconName="Heart" className="flex-1">
              Salvar
            </Button>
            <Button variant="outline" size="sm" iconName="Share2" className="flex-1">
              Compartilhar
            </Button>
          </div>
        </div>

        {/* Job Info */}
        <div className="text-xs text-muted-foreground pt-2 border-t border-border">
          <div className="flex items-center justify-between">
            <span>Publicado em: {new Date()?.toLocaleDateString('pt-BR')}</span>
            <span>ID: #JOB-{Math.random()?.toString(36)?.substr(2, 6)?.toUpperCase()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default JobPreview;