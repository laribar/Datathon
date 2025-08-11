import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';
import CandidateTable from './components/CandidateTable';
import FilterPanel from './components/FilterPanel';
import BulkActionsBar from './components/BulkActionsBar';
import CandidateModal from './components/CandidateModal';
import SearchBar from './components/SearchBar';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

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
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0,
    new: 0,
    interviewing: 0,
    approved: 0,
    rejected: 0,
    highMatch: 0
  });

  const fetchCandidates = async () => {
    setLoading(true);
    try {
      const queryParams = new URLSearchParams();

      if (searchTerm) queryParams.append('search', searchTerm);
      if (filters.status) queryParams.append('status', filters.status);
      if (filters.position) queryParams.append('position', filters.position);
      if (filters.scoreRange) {
        const [min, max] = filters.scoreRange.split('-').map(Number);
        if (!isNaN(min)) queryParams.append('min_score', min);
        if (!isNaN(max)) queryParams.append('max_score', max);
      }
      if (filters.dateFrom) queryParams.append('start_date', filters.dateFrom);
      if (filters.dateTo) queryParams.append('end_date', filters.dateTo);

      queryParams.append('order_by', 'date');
      queryParams.append('order_dir', 'desc');

      const url = `${API_BASE_URL}/candidates?${queryParams.toString()}`;
      console.log('[DEBUG] Buscando candidatos de:', url);

      const res = await fetch(url);
      if (!res.ok) throw new Error(`Erro ao buscar candidatos: ${res.status}`);

      const data = await res.json();
      console.log('[DEBUG] Resposta da API:', data);

      const candidatesList = data.candidates || data.data || data.results || [];
      const statsData = data.stats || {
        total: candidatesList.length,
        new: 0,
        interviewing: 0,
        approved: 0,
        rejected: 0,
        highMatch: 0
      };

      setCandidates(candidatesList);
      setStats(statsData);
    } catch (error) {
      console.error('[ERRO] Falha ao buscar candidatos:', error);
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCandidates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, searchTerm]);

  const handleSelectCandidate = (candidateId) => {
    setSelectedCandidates(prev =>
      prev.includes(candidateId)
        ? prev.filter(id => id !== candidateId)
        : [...prev, candidateId]
    );
  };

  const handleSelectAll = () => {
    if (selectedCandidates.length === candidates.length) {
      setSelectedCandidates([]);
    } else {
      setSelectedCandidates(candidates.map(candidate => candidate.id));
    }
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

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

  const handleViewProfile = (candidate) => {
    setSelectedCandidate(candidate);
    setIsModalOpen(true);
  };

  const handleScheduleInterview = () => {
    navigate('/video-interview-room');
  };

  const handleGenerateReport = () => {
    navigate('/interview-transcription-analysis');
  };

  const handleBulkStatusUpdate = (status) => {
    console.log('Atualizar status para:', status, 'candidatos:', selectedCandidates);
    setSelectedCandidates([]);
  };

  const handleBulkScheduleInterview = () => {
    navigate('/video-interview-room');
  };

  const handleBulkExport = () => {
    console.log('Exportando candidatos:', selectedCandidates);
  };

  const handleClearSelection = () => {
    setSelectedCandidates([]);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Gestão de Candidatos</h1>
              <p className="text-muted-foreground mt-2">
                Gerencie e avalie candidatos com análise de IA e ferramentas avançadas
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <Button variant="outline" iconName="Download" onClick={() => navigate('/recruitment-analytics-dashboard')}>
                Relatórios
              </Button>
              <Button variant="default" iconName="Plus" onClick={() => navigate('/job-posting-creation')}>
                Nova Vaga
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            <StatCard label="Total" value={stats.total} icon="Users" color="primary" />
            <StatCard label="Novos" value={stats.new} icon="UserPlus" color="blue-400" />
            <StatCard label="Entrevistando" value={stats.interviewing} icon="Video" color="warning" />
            <StatCard label="Aprovados" value={stats.approved} icon="CheckCircle" color="success" />
            <StatCard label="Rejeitados" value={stats.rejected} icon="XCircle" color="destructive" />
            <StatCard label="Alta Compatibilidade" value={stats.highMatch} icon="Target" color="success" />
          </div>

          <div className="mb-6">
            <SearchBar
              searchTerm={searchTerm}
              onSearchChange={setSearchTerm}
              placeholder="Buscar por nome, email ou posição..."
            />
          </div>
        </div>

        <FilterPanel
          filters={filters}
          onFilterChange={handleFilterChange}
          onClearFilters={handleClearFilters}
          isExpanded={isFilterExpanded}
          onToggleExpanded={() => setIsFilterExpanded(!isFilterExpanded)}
        />

        <BulkActionsBar
          selectedCount={selectedCandidates.length}
          onBulkStatusUpdate={handleBulkStatusUpdate}
          onBulkScheduleInterview={handleBulkScheduleInterview}
          onBulkExport={handleBulkExport}
          onClearSelection={handleClearSelection}
        />

        <div className="flex items-center justify-between mb-4">
          <p className="text-sm text-muted-foreground">
            Mostrando {candidates.length} candidatos
          </p>
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

        {loading ? (
          <p className="text-muted-foreground text-center">Carregando...</p>
        ) : (
          <CandidateTable
            candidates={candidates}
            selectedCandidates={selectedCandidates}
            onSelectCandidate={handleSelectCandidate}
            onSelectAll={handleSelectAll}
            onViewProfile={handleViewProfile}
            onScheduleInterview={handleScheduleInterview}
            onGenerateReport={handleGenerateReport}
          />
        )}

        {!loading && candidates.length === 0 && (
          <div className="text-center py-12">
            <Icon name="Users" size={48} className="text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-foreground mb-2">Nenhum candidato encontrado</h3>
            <p className="text-muted-foreground mb-4">Tente ajustar os filtros ou termos de busca</p>
            <Button variant="outline" onClick={handleClearFilters}>Limpar Filtros</Button>
          </div>
        )}

        <CandidateModal
          candidate={selectedCandidate}
          isOpen={isModalOpen}
          onClose={() => { setIsModalOpen(false); setSelectedCandidate(null); }}
          onScheduleInterview={handleScheduleInterview}
          onGenerateReport={handleGenerateReport}
        />
      </div>
    </div>
  );
};

const StatCard = ({ label, value, icon, color }) => (
  <div className="bg-card rounded-lg p-4 shadow-elevation-1">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-2xl font-bold" style={{ color: `var(--${color})` }}>{value}</p>
      </div>
      <Icon name={icon} className={`text-${color}`} size={24} />
    </div>
  </div>
);

export default CandidateManagement;
