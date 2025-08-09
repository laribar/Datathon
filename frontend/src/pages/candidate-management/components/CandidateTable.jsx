import React from 'react';

import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';
import CandidateScoreChart from './CandidateScoreChart';
import StatusBadge from './StatusBadge';

const CandidateTable = ({ 
  candidates, 
  selectedCandidates, 
  onSelectCandidate, 
  onSelectAll, 
  onViewProfile, 
  onScheduleInterview, 
  onGenerateReport 
}) => {
  const isAllSelected = candidates?.length > 0 && selectedCandidates?.length === candidates?.length;
  const isIndeterminate = selectedCandidates?.length > 0 && selectedCandidates?.length < candidates?.length;

  return (
    <div className="bg-card rounded-lg shadow-elevation-1 overflow-hidden">
      {/* Table Header */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-muted/20 border-b border-border">
            <tr>
              <th className="w-12 px-4 py-3 text-left">
                <input
                  type="checkbox"
                  checked={isAllSelected}
                  ref={input => {
                    if (input) input.indeterminate = isIndeterminate;
                  }}
                  onChange={onSelectAll}
                  className="w-4 h-4 text-primary bg-input border-border rounded focus:ring-primary focus:ring-2"
                />
              </th>
              <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                Candidato
              </th>
              <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                Posição
              </th>
              <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                Data de Aplicação
              </th>
              <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                Compatibilidade IA
              </th>
              <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                Status
              </th>
              <th className="px-4 py-3 text-right text-sm font-medium text-muted-foreground">
                Ações
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {candidates?.map((candidate) => (
              <tr 
                key={candidate?.id} 
                className="hover:bg-muted/10 transition-smooth"
              >
                <td className="px-4 py-4">
                  <input
                    type="checkbox"
                    checked={selectedCandidates?.includes(candidate?.id)}
                    onChange={() => onSelectCandidate(candidate?.id)}
                    className="w-4 h-4 text-primary bg-input border-border rounded focus:ring-primary focus:ring-2"
                  />
                </td>
                <td className="px-4 py-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-full overflow-hidden bg-muted flex-shrink-0">
                      <Image
                        src={candidate?.avatar}
                        alt={candidate?.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div>
                      <button
                        onClick={() => onViewProfile(candidate)}
                        className="font-medium text-foreground hover:text-primary transition-smooth"
                      >
                        {candidate?.name}
                      </button>
                      <p className="text-sm text-muted-foreground">{candidate?.email}</p>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-4">
                  <div className="text-sm text-foreground">{candidate?.position}</div>
                  <div className="text-xs text-muted-foreground">{candidate?.department}</div>
                </td>
                <td className="px-4 py-4 text-sm text-muted-foreground">
                  {candidate?.applicationDate}
                </td>
                <td className="px-4 py-4">
                  <CandidateScoreChart score={candidate?.compatibilityScore} size={40} />
                </td>
                <td className="px-4 py-4">
                  <StatusBadge status={candidate?.status} />
                </td>
                <td className="px-4 py-4">
                  <div className="flex items-center justify-end space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="Eye"
                      onClick={() => onViewProfile(candidate)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      Ver
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="Calendar"
                      onClick={() => onScheduleInterview(candidate)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      Agendar
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      iconName="FileText"
                      onClick={() => onGenerateReport(candidate)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      Relatório
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Mobile Card Layout - Hidden on desktop */}
      <div className="lg:hidden">
        {candidates?.map((candidate) => (
          <div key={candidate?.id} className="p-4 border-b border-border last:border-b-0">
            <div className="flex items-start space-x-3">
              <input
                type="checkbox"
                checked={selectedCandidates?.includes(candidate?.id)}
                onChange={() => onSelectCandidate(candidate?.id)}
                className="w-4 h-4 text-primary bg-input border-border rounded focus:ring-primary focus:ring-2 mt-1"
              />
              <div className="w-12 h-12 rounded-full overflow-hidden bg-muted flex-shrink-0">
                <Image
                  src={candidate?.avatar}
                  alt={candidate?.name}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-2">
                  <button
                    onClick={() => onViewProfile(candidate)}
                    className="font-medium text-foreground hover:text-primary transition-smooth truncate"
                  >
                    {candidate?.name}
                  </button>
                  <CandidateScoreChart score={candidate?.compatibilityScore} size={32} />
                </div>
                <p className="text-sm text-muted-foreground mb-1">{candidate?.position}</p>
                <p className="text-xs text-muted-foreground mb-2">{candidate?.applicationDate}</p>
                <div className="flex items-center justify-between">
                  <StatusBadge status={candidate?.status} />
                  <div className="flex space-x-1">
                    <Button
                      variant="ghost"
                      size="xs"
                      iconName="Eye"
                      onClick={() => onViewProfile(candidate)}
                    />
                    <Button
                      variant="ghost"
                      size="xs"
                      iconName="Calendar"
                      onClick={() => onScheduleInterview(candidate)}
                    />
                    <Button
                      variant="ghost"
                      size="xs"
                      iconName="FileText"
                      onClick={() => onGenerateReport(candidate)}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CandidateTable;