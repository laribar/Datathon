import React, { useState } from 'react';
import { Helmet } from 'react-helmet';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import JobForm from './components/JobForm';
import RichTextEditor from './components/RichTextEditor';
import JobPreview from './components/JobPreview';
import Button from '../../components/ui/Button';


const JobPostingCreation = () => {
  const [formData, setFormData] = useState({
    title: '',
    department: '',
    location: '',
    employmentType: '',
    experience: '',
    salaryMin: '',
    salaryMax: '',
    salaryNegotiable: false,
    urgent: false,
    description: '',
    technicalSkills: '',
    softSkills: '',
    education: '',
    benefits: ''
  });

  const [isLoading, setIsLoading] = useState(false);
  const [showMobilePreview, setShowMobilePreview] = useState(false);

  const handleSave = async () => {
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      console.log('Rascunho salvo:', formData);
      setIsLoading(false);
      
      // Show success notification
      alert('Rascunho salvo com sucesso!');
    }, 1500);
  };

  const handlePublish = async () => {
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      console.log('Vaga publicada:', formData);
      setIsLoading(false);
      
      // Show success notification
      alert('Vaga publicada com sucesso!');
    }, 2000);
  };

  const handleDescriptionChange = (content) => {
    setFormData(prev => ({
      ...prev,
      description: content
    }));
  };

  return (
    <>
      <Helmet>
        <title>Criar Nova Vaga - RecrutaAI</title>
        <meta name="description" content="Crie e publique novas oportunidades de trabalho com nossa plataforma de recrutamento inteligente" />
      </Helmet>
      <div className="flex min-h-screen bg-background">
        {/* Sidebar - Hidden on mobile */}
        <div className="hidden lg:block">
          <Sidebar />
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          <Header />
          
          <main className="flex-1 p-6">
            <div className="max-w-7xl mx-auto">
              {/* Mobile Preview Toggle */}
              <div className="lg:hidden mb-4">
                <Button
                  variant="outline"
                  onClick={() => setShowMobilePreview(!showMobilePreview)}
                  iconName={showMobilePreview ? "Edit" : "Eye"}
                  iconPosition="left"
                  fullWidth
                >
                  {showMobilePreview ? 'Editar Vaga' : 'Ver Pré-visualização'}
                </Button>
              </div>

              {/* Desktop Layout */}
              <div className="hidden lg:grid lg:grid-cols-5 gap-6">
                {/* Form Section - 60% */}
                <div className="lg:col-span-3 space-y-6">
                  <JobForm
                    formData={formData}
                    setFormData={setFormData}
                    onSave={handleSave}
                    onPublish={handlePublish}
                    isLoading={isLoading}
                  />
                  
                  {/* Rich Text Editor */}
                  <div className="bg-card rounded-lg shadow-elevation-1 p-6">
                    <RichTextEditor
                      value={formData?.description}
                      onChange={handleDescriptionChange}
                      error={!formData?.description ? 'Descrição da vaga é obrigatória' : ''}
                    />
                  </div>
                </div>

                {/* Preview Section - 40% */}
                <div className="lg:col-span-2">
                  <JobPreview formData={formData} />
                </div>
              </div>

              {/* Mobile Layout */}
              <div className="lg:hidden">
                {!showMobilePreview ? (
                  <div className="space-y-6">
                    <JobForm
                      formData={formData}
                      setFormData={setFormData}
                      onSave={handleSave}
                      onPublish={handlePublish}
                      isLoading={isLoading}
                    />
                    
                    <div className="bg-card rounded-lg shadow-elevation-1 p-6">
                      <RichTextEditor
                        value={formData?.description}
                        onChange={handleDescriptionChange}
                        error={!formData?.description ? 'Descrição da vaga é obrigatória' : ''}
                      />
                    </div>
                  </div>
                ) : (
                  <JobPreview formData={formData} />
                )}
              </div>

              {/* Fixed Action Buttons - Desktop */}
              <div className="hidden lg:block fixed bottom-6 right-6 z-50">
                <div className="flex flex-col gap-3">
                  <Button
                    variant="outline"
                    size="lg"
                    onClick={handleSave}
                    iconName="Save"
                    iconPosition="left"
                    disabled={isLoading}
                    className="shadow-elevation-2"
                  >
                    Salvar Rascunho
                  </Button>
                  <Button
                    variant="default"
                    size="lg"
                    onClick={handlePublish}
                    iconName="Send"
                    iconPosition="left"
                    loading={isLoading}
                    className="shadow-elevation-2"
                  >
                    Publicar Vaga
                  </Button>
                </div>
              </div>

              {/* Fixed Action Buttons - Mobile */}
              <div className="lg:hidden fixed bottom-4 left-4 right-4 z-50">
                <div className="bg-card rounded-lg shadow-elevation-3 p-4 border border-border">
                  <div className="flex gap-3">
                    <Button
                      variant="outline"
                      onClick={handleSave}
                      iconName="Save"
                      iconPosition="left"
                      disabled={isLoading}
                      className="flex-1"
                    >
                      Salvar
                    </Button>
                    <Button
                      variant="default"
                      onClick={handlePublish}
                      iconName="Send"
                      iconPosition="left"
                      loading={isLoading}
                      className="flex-1"
                    >
                      Publicar
                    </Button>
                  </div>
                </div>
              </div>

              {/* Bottom Spacing for Fixed Buttons */}
              <div className="h-24 lg:h-0"></div>
            </div>
          </main>
        </div>
      </div>
    </>
  );
};

export default JobPostingCreation;