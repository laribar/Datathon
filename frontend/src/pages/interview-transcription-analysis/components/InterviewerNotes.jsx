import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const InterviewerNotes = ({ notes, onAddNote, onUpdateNote, onDeleteNote }) => {
  const [newNote, setNewNote] = useState('');
  const [editingNote, setEditingNote] = useState(null);
  const [editText, setEditText] = useState('');

  const handleAddNote = () => {
    if (newNote?.trim()) {
      onAddNote({
        id: Date.now(),
        text: newNote,
        author: 'Você',
        timestamp: new Date(),
        private: true
      });
      setNewNote('');
    }
  };

  const handleEditNote = (note) => {
    setEditingNote(note?.id);
    setEditText(note?.text);
  };

  const handleSaveEdit = () => {
    if (editText?.trim()) {
      onUpdateNote(editingNote, editText);
      setEditingNote(null);
      setEditText('');
    }
  };

  const handleCancelEdit = () => {
    setEditingNote(null);
    setEditText('');
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp)?.toLocaleString('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="bg-card rounded-lg border border-border p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-foreground">Anotações dos Entrevistadores</h3>
        <Icon name="StickyNote" size={20} className="text-muted-foreground" />
      </div>
      {/* Add New Note */}
      <div className="mb-6 p-4 bg-muted/30 rounded-lg">
        <textarea
          value={newNote}
          onChange={(e) => setNewNote(e?.target?.value)}
          placeholder="Adicionar nova anotação..."
          className="w-full h-20 p-3 bg-input border border-border rounded-lg text-foreground placeholder-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary"
        />
        <div className="flex justify-between items-center mt-3">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="private-note"
              className="rounded border-border text-primary focus:ring-primary"
            />
            <label htmlFor="private-note" className="text-sm text-muted-foreground">
              Anotação privada
            </label>
          </div>
          <Button
            onClick={handleAddNote}
            disabled={!newNote?.trim()}
            size="sm"
            iconName="Plus"
          >
            Adicionar
          </Button>
        </div>
      </div>
      {/* Notes List */}
      <div className="space-y-4">
        {notes?.map((note) => (
          <div key={note?.id} className="border border-border rounded-lg p-4">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center">
                  <span className="text-xs font-medium text-primary">
                    {note?.author?.charAt(0)?.toUpperCase()}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-foreground">{note?.author}</span>
                  {note?.private && (
                    <span className="ml-2 px-2 py-1 bg-warning/20 text-warning text-xs rounded-full">
                      Privada
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">
                  {formatTimestamp(note?.timestamp)}
                </span>
                <div className="flex gap-1">
                  <button
                    onClick={() => handleEditNote(note)}
                    className="p-1 text-muted-foreground hover:text-primary transition-colors"
                  >
                    <Icon name="Edit2" size={14} />
                  </button>
                  <button
                    onClick={() => onDeleteNote(note?.id)}
                    className="p-1 text-muted-foreground hover:text-destructive transition-colors"
                  >
                    <Icon name="Trash2" size={14} />
                  </button>
                </div>
              </div>
            </div>

            {editingNote === note?.id ? (
              <div className="space-y-3">
                <textarea
                  value={editText}
                  onChange={(e) => setEditText(e?.target?.value)}
                  className="w-full h-20 p-3 bg-input border border-border rounded-lg text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-primary"
                />
                <div className="flex gap-2">
                  <Button size="sm" onClick={handleSaveEdit}>
                    Salvar
                  </Button>
                  <Button size="sm" variant="outline" onClick={handleCancelEdit}>
                    Cancelar
                  </Button>
                </div>
              </div>
            ) : (
              <p className="text-foreground leading-relaxed">{note?.text}</p>
            )}
          </div>
        ))}

        {notes?.length === 0 && (
          <div className="text-center py-8">
            <Icon name="StickyNote" size={48} className="mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">Nenhuma anotação adicionada ainda</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default InterviewerNotes;