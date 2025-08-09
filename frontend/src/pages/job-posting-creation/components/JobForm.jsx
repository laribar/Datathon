import React, { useState } from 'react';
import Input from '../../../components/ui/Input';
import Select from '../../../components/ui/Select';
import Button from '../../../components/ui/Button';
import { Checkbox } from '../../../components/ui/Checkbox';
import Icon from '../../../components/AppIcon';

const JobForm = ({ formData, setFormData, onSave, onPublish, isLoading }) => {
  const [expandedSections, setExpandedSections] = useState({
    basic: true,
    details: true,
    requirements: true,
    benefits: false
  });

  const [errors, setErrors] = useState({});

  const departmentOptions = [
    { value: 'tecnologia', label: 'Tecnologia' },
    { value: 'marketing', label: 'Marketing' },
    { value: 'vendas', label: 'Vendas' },
    { value: 'rh', label: 'Recursos Humanos' },
    { value: 'financeiro', label: 'Financeiro' },
    { value: 'operacoes', label: 'Operações' }
  ];

  const locationOptions = [
    { value: 'sao-paulo', label: 'São Paulo, SP' },
    { value: 'rio-janeiro', label: 'Rio de Janeiro, RJ' },
    { value: 'belo-horizonte', label: 'Belo Horizonte, MG' },
    { value: 'brasilia', label: 'Brasília, DF' },
    { value: 'remoto', label: 'Remoto' },
    { value: 'hibrido', label: 'Híbrido' }
  ];

  const employmentTypeOptions = [
    { value: 'clt', label: 'CLT' },
    { value: 'pj', label: 'PJ' },
    { value: 'estagio', label: 'Estágio' },
    { value: 'freelancer', label: 'Freelancer' },
    { value: 'temporario', label: 'Temporário' }
  ];

  const experienceOptions = [
    { value: 'junior', label: 'Júnior (0-2 anos)' },
    { value: 'pleno', label: 'Pleno (2-5 anos)' },
    { value: 'senior', label: 'Sênior (5+ anos)' },
    { value: 'especialista', label: 'Especialista (8+ anos)' }
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error when user starts typing
    if (errors?.[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev?.[section]
    }));
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData?.title?.trim()) {
      newErrors.title = 'Título da vaga é obrigatório';
    }
    
    if (!formData?.department) {
      newErrors.department = 'Departamento é obrigatório';
    }
    
    if (!formData?.location) {
      newErrors.location = 'Localização é obrigatória';
    }
    
    if (!formData?.employmentType) {
      newErrors.employmentType = 'Tipo de contratação é obrigatório';
    }
    
    if (!formData?.description?.trim()) {
      newErrors.description = 'Descrição da vaga é obrigatória';
    }

    setErrors(newErrors);
    return Object.keys(newErrors)?.length === 0;
  };

  const handlePublish = () => {
    if (validateForm()) {
      onPublish();
    }
  };

  const handleSaveDraft = () => {
    onSave();
  };

  return (
    <div className="bg-card rounded-lg shadow-elevation-1 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-foreground">Criar Nova Vaga</h2>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleSaveDraft}
            iconName="Save"
            iconPosition="left"
            disabled={isLoading}
          >
            Salvar Rascunho
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={handlePublish}
            iconName="Send"
            iconPosition="left"
            loading={isLoading}
          >
            Publicar Vaga
          </Button>
        </div>
      </div>
      <div className="space-y-6">
        {/* Basic Information Section */}
        <div className="border border-border rounded-lg">
          <button
            onClick={() => toggleSection('basic')}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-muted/50 transition-smooth"
          >
            <h3 className="font-medium text-foreground">Informações Básicas</h3>
            <Icon 
              name={expandedSections?.basic ? "ChevronUp" : "ChevronDown"} 
              size={20} 
            />
          </button>
          
          {expandedSections?.basic && (
            <div className="p-4 pt-0 space-y-4">
              <Input
                label="Título da Vaga"
                type="text"
                placeholder="Ex: Desenvolvedor Full Stack Sênior"
                value={formData?.title || ''}
                onChange={(e) => handleInputChange('title', e?.target?.value)}
                error={errors?.title}
                required
              />
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Select
                  label="Departamento"
                  placeholder="Selecione o departamento"
                  options={departmentOptions}
                  value={formData?.department || ''}
                  onChange={(value) => handleInputChange('department', value)}
                  error={errors?.department}
                  required
                />
                
                <Select
                  label="Localização"
                  placeholder="Selecione a localização"
                  options={locationOptions}
                  value={formData?.location || ''}
                  onChange={(value) => handleInputChange('location', value)}
                  error={errors?.location}
                  required
                />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Select
                  label="Tipo de Contratação"
                  placeholder="Selecione o tipo"
                  options={employmentTypeOptions}
                  value={formData?.employmentType || ''}
                  onChange={(value) => handleInputChange('employmentType', value)}
                  error={errors?.employmentType}
                  required
                />
                
                <Select
                  label="Nível de Experiência"
                  placeholder="Selecione o nível"
                  options={experienceOptions}
                  value={formData?.experience || ''}
                  onChange={(value) => handleInputChange('experience', value)}
                />
              </div>
            </div>
          )}
        </div>

        {/* Salary and Details Section */}
        <div className="border border-border rounded-lg">
          <button
            onClick={() => toggleSection('details')}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-muted/50 transition-smooth"
          >
            <h3 className="font-medium text-foreground">Detalhes da Vaga</h3>
            <Icon 
              name={expandedSections?.details ? "ChevronUp" : "ChevronDown"} 
              size={20} 
            />
          </button>
          
          {expandedSections?.details && (
            <div className="p-4 pt-0 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  label="Salário Mínimo (R$)"
                  type="number"
                  placeholder="5000"
                  value={formData?.salaryMin || ''}
                  onChange={(e) => handleInputChange('salaryMin', e?.target?.value)}
                />
                
                <Input
                  label="Salário Máximo (R$)"
                  type="number"
                  placeholder="8000"
                  value={formData?.salaryMax || ''}
                  onChange={(e) => handleInputChange('salaryMax', e?.target?.value)}
                />
              </div>
              
              <div className="flex items-center gap-4">
                <Checkbox
                  label="Salário a combinar"
                  checked={formData?.salaryNegotiable || false}
                  onChange={(e) => handleInputChange('salaryNegotiable', e?.target?.checked)}
                />
                
                <Checkbox
                  label="Vaga urgente"
                  checked={formData?.urgent || false}
                  onChange={(e) => handleInputChange('urgent', e?.target?.checked)}
                />
              </div>
            </div>
          )}
        </div>

        {/* Requirements Section */}
        <div className="border border-border rounded-lg">
          <button
            onClick={() => toggleSection('requirements')}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-muted/50 transition-smooth"
          >
            <h3 className="font-medium text-foreground">Requisitos</h3>
            <Icon 
              name={expandedSections?.requirements ? "ChevronUp" : "ChevronDown"} 
              size={20} 
            />
          </button>
          
          {expandedSections?.requirements && (
            <div className="p-4 pt-0 space-y-4">
              <Input
                label="Habilidades Técnicas"
                type="text"
                placeholder="React, Node.js, Python, SQL (separadas por vírgula)"
                value={formData?.technicalSkills || ''}
                onChange={(e) => handleInputChange('technicalSkills', e?.target?.value)}
                description="Digite as habilidades separadas por vírgula"
              />
              
              <Input
                label="Habilidades Comportamentais"
                type="text"
                placeholder="Liderança, Comunicação, Trabalho em equipe"
                value={formData?.softSkills || ''}
                onChange={(e) => handleInputChange('softSkills', e?.target?.value)}
                description="Digite as habilidades separadas por vírgula"
              />
              
              <Input
                label="Formação Acadêmica"
                type="text"
                placeholder="Ensino Superior em Ciência da Computação ou áreas afins"
                value={formData?.education || ''}
                onChange={(e) => handleInputChange('education', e?.target?.value)}
              />
            </div>
          )}
        </div>

        {/* Benefits Section */}
        <div className="border border-border rounded-lg">
          <button
            onClick={() => toggleSection('benefits')}
            className="w-full flex items-center justify-between p-4 text-left hover:bg-muted/50 transition-smooth"
          >
            <h3 className="font-medium text-foreground">Benefícios</h3>
            <Icon 
              name={expandedSections?.benefits ? "ChevronUp" : "ChevronDown"} 
              size={20} 
            />
          </button>
          
          {expandedSections?.benefits && (
            <div className="p-4 pt-0 space-y-4">
              <Input
                label="Benefícios Oferecidos"
                type="text"
                placeholder="Vale refeição, Plano de saúde, Home office"
                value={formData?.benefits || ''}
                onChange={(e) => handleInputChange('benefits', e?.target?.value)}
                description="Digite os benefícios separados por vírgula"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default JobForm;