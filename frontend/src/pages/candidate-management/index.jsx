import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';
import CandidateTable from './components/CandidateTable';
import FilterPanel from './components/FilterPanel';
import BulkActionsBar from './components/BulkActionsBar';
import CandidateModal from './components/CandidateModal';
import SearchBar from './components/SearchBar';

const CandidateManagement = () => {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCandidates, setSelectedCandidates] = useState([]);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isFilterExpanded, setIsFilterExpanded] = useState(false);
  const [filters, setFilters] = useState({
    status: '',
    position: '',
    scoreRange: '',
    dateFrom: '',
    dateTo: ''
  });

  // Mock candidates data
  const mockCandidates = [
    {
      id: 1,
      name: "Ana Silva Santos",
      email: "ana.silva@email.com",
      position: "Desenvolvedor Frontend",
      department: "Tecnologia",
      applicationDate: "05/09/2025",
      compatibilityScore: 92,
      status: "novo",
      avatar: "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150&h=150&fit=crop&crop=face",
      experience: "4 anos",
      location: "São Paulo, SP"
    },
    {
      id: 2,
      name: "Carlos Eduardo Lima",
      email: "carlos.lima@email.com",
      position: "Desenvolvedor Backend",
      department: "Tecnologia",
      applicationDate: "04/09/2025",
      compatibilityScore: 87,
      status: "entrevistando",
      avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face",
      experience: "6 anos",
      location: "Rio de Janeiro, RJ"
    },
    {
      id: 3,
      name: "Mariana Costa Oliveira",
      email: "mariana.costa@email.com",
      position: "Designer UI/UX",
      department: "Design",
      applicationDate: "03/09/2025",
      compatibilityScore: 78,
      status: "aprovado",
      avatar: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face",
      experience: "3 anos",
      location: "Belo Horizonte, MG"
    },
    {
      id: 4,
      name: "Roberto Ferreira Souza",
      email: "roberto.souza@email.com",
      position: "Desenvolvedor Full Stack",
      department: "Tecnologia",
      applicationDate: "02/09/2025",
      compatibilityScore: 65,
      status: "rejeitado",
      avatar: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face",
      experience: "5 anos",
      location: "Porto Alegre, RS"
    },
    {
      id: 5,
      name: "Juliana Pereira Santos",
      email: "juliana.pereira@email.com",
      position: "Gerente de Produto",
      department: "Produto",
      applicationDate: "01/09/2025",
      compatibilityScore: 89,
      status: "entrevistando",
      avatar: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&h=150&fit=crop&crop=face",
      experience: "7 anos",
      location: "São Paulo, SP"
    },
    {
      id: 6,
      name: "Fernando Alves Costa",
      email: "fernando.alves@email.com",
      position: "Analista de Dados",
      department: "Analytics",
      applicationDate: "31/08/2025",
      compatibilityScore: 73,
      status: "novo",
      avatar: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&h=150&fit=crop&crop=face",
      experience: "4 anos",
      location: "Brasília, DF"
    },
    {
      id: 7,
      name: "Camila Rodrigues Silva",
      email: "camila.rodrigues@email.com",
      position: "Engenheiro DevOps",
      department: "Infraestrutura",
      applicationDate: "30/08/2025",
      compatibilityScore: 81,
      status: "novo",
      avatar: "https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?w=150&h=150&fit=crop&crop=face",
      experience: "5 anos",
      location: "Florianópolis, SC"
    },
    {
      id: 8,
      name: "Lucas Martins Oliveira",
      email: "lucas.martins@email.com",
      position: "Desenvolvedor Frontend",
      department: "Tecnologia",
      applicationDate: "29/08/2025",
      compatibilityScore: 76,
      status: "entrevistando",
      avatar: "https://images.unsplash.com/photo-1519244703995-f4e0f30006d5?w=150&h=150&fit=crop&crop=face",
      experience: "3 anos",
      location: "Curitiba, PR"
    }
  ];

  // Filter candidates based on search term and filters
  const filteredCandidates = useMemo(() => {
    let filtered = mockCandidates;

    // Search filter
    if (searchTerm) {
      filtered = filtered?.filter(candidate =>
        candidate?.name?.toLowerCase()?.includes(searchTerm?.toLowerCase()) ||
        candidate?.email?.toLowerCase()?.includes(searchTerm?.toLowerCase()) ||
        candidate?.position?.toLowerCase()?.includes(searchTerm?.toLowerCase())
      );
    }

    // Status filter
    if (filters?.status) {
      filtered = filtered?.filter(candidate => candidate?.status === filters?.status);
    }

    // Position filter
    if (filters?.position) {
      filtered = filtered?.filter(candidate => 
        candidate?.position?.toLowerCase()?.includes(filters?.position?.toLowerCase())
      );
    }

    // Score range filter
    if (filters?.scoreRange) {
      const [min, max] = filters?.scoreRange?.split('-')?.map(Number);
      filtered = filtered?.filter(candidate => 
        candidate?.compatibilityScore >= min && candidate?.compatibilityScore <= max
      );
    }

    // Date range filter
    if (filters?.dateFrom || filters?.dateTo) {
      filtered = filtered?.filter(candidate => {
        const candidateDate = new Date(candidate.applicationDate.split('/').reverse().join('-'));
        const fromDate = filters?.dateFrom ? new Date(filters.dateFrom) : null;
        const toDate = filters?.dateTo ? new Date(filters.dateTo) : null;

        if (fromDate && candidateDate < fromDate) return false;
        if (toDate && candidateDate > toDate) return false;
        return true;
      });
    }

    return filtered;
  }, [searchTerm, filters]);

  // Handle candidate selection
  const handleSelectCandidate = (candidateId) => {
    setSelectedCandidates(prev =>
      prev?.includes(candidateId)
        ? prev?.filter(id => id !== candidateId)
        : [...prev, candidateId]
    );
  };

  // Handle select all candidates
  const handleSelectAll = () => {
    if (selectedCandidates?.length === filteredCandidates?.length) {
      setSelectedCandidates([]);
    } else {
      setSelectedCandidates(filteredCandidates?.map(candidate => candidate?.id));
    }
  };

  // Handle filter changes
  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  // Clear all filters
  const handleClearFilters = () => {
    setFilters({
      status: '',
      position: '',
      scoreRange: '',
      dateFrom: '',
      dateTo: ''
    });
    setSearchTerm('');
  };

  // Handle view candidate profile
  const handleViewProfile = (candidate) => {
    setSelectedCandidate(candidate);
    setIsModalOpen(true);
  };

  // Handle schedule interview
  const handleScheduleInterview = (candidate) => {
    console.log('Scheduling interview for:', candidate?.name);
    // Navigate to video interview room or show scheduling modal
    navigate('/video-interview-room');
  };

  // Handle generate report
  const handleGenerateReport = (candidate) => {
    console.log('Generating report for:', candidate?.name);
    // Navigate to interview transcription analysis
    navigate('/interview-transcription-analysis');
  };

  // Handle bulk actions
  const handleBulkStatusUpdate = (status) => {
    console.log('Updating status to:', status, 'for candidates:', selectedCandidates);
    // Update status for selected candidates
    setSelectedCandidates([]);
  };

  const handleBulkScheduleInterview = () => {
    console.log('Scheduling interviews for candidates:', selectedCandidates);
    navigate('/video-interview-room');
  };

  const handleBulkExport = () => {
    console.log('Exporting candidates:', selectedCandidates);
    // Export selected candidates data
  };

  const handleClearSelection = () => {
    setSelectedCandidates([]);
  };

  // Statistics
  const stats = {
    total: mockCandidates?.length,
    new: mockCandidates?.filter(c => c?.status === 'novo')?.length,
    interviewing: mockCandidates?.filter(c => c?.status === 'entrevistando')?.length,
    approved: mockCandidates?.filter(c => c?.status === 'aprovado')?.length,
    rejected: mockCandidates?.filter(c => c?.status === 'rejeitado')?.length,
    highMatch: mockCandidates?.filter(c => c?.compatibilityScore >= 80)?.length
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Gestão de Candidatos</h1>
              <p className="text-muted-foreground mt-2">
                Gerencie e avalie candidatos com análise de IA e ferramentas avançadas
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <Button
                variant="outline"
                iconName="Download"
                onClick={() => navigate('/recruitment-analytics-dashboard')}
              >
                Relatórios
              </Button>
              <Button
                variant="default"
                iconName="Plus"
                onClick={() => navigate('/job-posting-creation')}
              >
                Nova Vaga
              </Button>
            </div>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            <div className="bg-card rounded-lg p-4 shadow-elevation-1">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Total</p>
                  <p className="text-2xl font-bold text-foreground">{stats?.total}</p>
                </div>
                <Icon name="Users" className="text-primary" size={24} />
              </div>
            </div>
            <div className="bg-card rounded-lg p-4 shadow-elevation-1">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Novos</p>
                  <p className="text-2xl font-bold text-blue-400">{stats?.new}</p>
                </div>
                <Icon name="UserPlus" className="text-blue-400" size={24} />
              </div>
            </div>
            <div className="bg-card rounded-lg p-4 shadow-elevation-1">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Entrevistando</p>
                  <p className="text-2xl font-bold text-warning">{stats?.interviewing}</p>
                </div>
                <Icon name="Video" className="text-warning" size={24} />
              </div>
            </div>
            <div className="bg-card rounded-lg p-4 shadow-elevation-1">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Aprovados</p>
                  <p className="text-2xl font-bold text-success">{stats?.approved}</p>
                </div>
                <Icon name="CheckCircle" className="text-success" size={24} />
              </div>
            </div>
            <div className="bg-card rounded-lg p-4 shadow-elevation-1">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Rejeitados</p>
                  <p className="text-2xl font-bold text-destructive">{stats?.rejected}</p>
                </div>
                <Icon name="XCircle" className="text-destructive" size={24} />
              </div>
            </div>
            <div className="bg-card rounded-lg p-4 shadow-elevation-1">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Alta Compatibilidade</p>
                  <p className="text-2xl font-bold text-success">{stats?.highMatch}</p>
                </div>
                <Icon name="Target" className="text-success" size={24} />
              </div>
            </div>
          </div>

          {/* Search Bar */}
          <div className="mb-6">
            <SearchBar
              searchTerm={searchTerm}
              onSearchChange={setSearchTerm}
              placeholder="Buscar por nome, email ou posição..."
            />
          </div>
        </div>

        {/* Filters */}
        <FilterPanel
          filters={filters}
          onFilterChange={handleFilterChange}
          onClearFilters={handleClearFilters}
          isExpanded={isFilterExpanded}
          onToggleExpanded={() => setIsFilterExpanded(!isFilterExpanded)}
        />

        {/* Bulk Actions */}
        <BulkActionsBar
          selectedCount={selectedCandidates?.length}
          onBulkStatusUpdate={handleBulkStatusUpdate}
          onBulkScheduleInterview={handleBulkScheduleInterview}
          onBulkExport={handleBulkExport}
          onClearSelection={handleClearSelection}
        />

        {/* Results Info */}
        <div className="flex items-center justify-between mb-4">
          <p className="text-sm text-muted-foreground">
            Mostrando {filteredCandidates?.length} de {mockCandidates?.length} candidatos
          </p>
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="sm"
              iconName="RotateCcw"
              onClick={handleClearFilters}
              className="text-muted-foreground hover:text-foreground"
            >
              Limpar Filtros
            </Button>
          </div>
        </div>

        {/* Candidates Table */}
        <CandidateTable
          candidates={filteredCandidates}
          selectedCandidates={selectedCandidates}
          onSelectCandidate={handleSelectCandidate}
          onSelectAll={handleSelectAll}
          onViewProfile={handleViewProfile}
          onScheduleInterview={handleScheduleInterview}
          onGenerateReport={handleGenerateReport}
        />

        {/* Empty State */}
        {filteredCandidates?.length === 0 && (
          <div className="text-center py-12">
            <Icon name="Users" size={48} className="text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">
              Nenhum candidato encontrado
            </h3>
            <p className="text-muted-foreground mb-4">
              Tente ajustar os filtros ou termos de busca
            </p>
            <Button
              variant="outline"
              onClick={handleClearFilters}
            >
              Limpar Filtros
            </Button>
          </div>
        )}

        {/* Candidate Modal */}
        <CandidateModal
          candidate={selectedCandidate}
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setSelectedCandidate(null);
          }}
          onScheduleInterview={handleScheduleInterview}
          onGenerateReport={handleGenerateReport}
        />
      </div>
    </div>
  );
};

export default CandidateManagement;