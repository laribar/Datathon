import React, { useState, useRef } from 'react';
import Button from '../../../components/ui/Button';


const RichTextEditor = ({ value, onChange, error }) => {
  const [activeFormats, setActiveFormats] = useState(new Set());
  const editorRef = useRef(null);

  const formatButtons = [
    { name: 'bold', icon: 'Bold', command: 'bold' },
    { name: 'italic', icon: 'Italic', command: 'italic' },
    { name: 'underline', icon: 'Underline', command: 'underline' },
    { name: 'insertUnorderedList', icon: 'List', command: 'insertUnorderedList' },
    { name: 'insertOrderedList', icon: 'ListOrdered', command: 'insertOrderedList' },
    { name: 'createLink', icon: 'Link', command: 'createLink' }
  ];

  const handleFormat = (command) => {
    if (command === 'createLink') {
      const url = prompt('Digite a URL do link:');
      if (url) {
        document.execCommand(command, false, url);
      }
    } else {
      document.execCommand(command, false, null);
    }
    
    editorRef?.current?.focus();
    updateActiveFormats();
  };

  const updateActiveFormats = () => {
    const formats = new Set();
    
    if (document.queryCommandState('bold')) formats?.add('bold');
    if (document.queryCommandState('italic')) formats?.add('italic');
    if (document.queryCommandState('underline')) formats?.add('underline');
    if (document.queryCommandState('insertUnorderedList')) formats?.add('insertUnorderedList');
    if (document.queryCommandState('insertOrderedList')) formats?.add('insertOrderedList');
    
    setActiveFormats(formats);
  };

  const handleInput = () => {
    const content = editorRef?.current?.innerHTML || '';
    onChange(content);
    updateActiveFormats();
  };

  const handleKeyDown = (e) => {
    if (e?.key === 'Tab') {
      e?.preventDefault();
      document.execCommand('insertHTML', false, '&nbsp;&nbsp;&nbsp;&nbsp;');
    }
  };

  const insertTemplate = () => {
    const template = `<h3>Sobre a Vaga</h3>
<p>Descreva aqui os principais objetivos e responsabilidades da posição.</p>

<h3>Responsabilidades</h3>
<ul>
<li>Responsabilidade 1</li>
<li>Responsabilidade 2</li>
<li>Responsabilidade 3</li>
</ul>

<h3>Requisitos</h3>
<ul>
<li>Requisito 1</li>
<li>Requisito 2</li>
<li>Requisito 3</li>
</ul>

<h3>Diferenciais</h3>
<ul>
<li>Diferencial 1</li>
<li>Diferencial 2</li>
</ul>`;

    editorRef.current.innerHTML = template;
    onChange(template);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-foreground">
          Descrição da Vaga *
        </label>
        <Button
          variant="ghost"
          size="sm"
          onClick={insertTemplate}
          iconName="FileText"
          iconPosition="left"
        >
          Usar Template
        </Button>
      </div>
      {/* Toolbar */}
      <div className="flex items-center gap-1 p-2 bg-muted/30 rounded-t-lg border border-border border-b-0">
        {formatButtons?.map((button) => (
          <Button
            key={button?.name}
            variant={activeFormats?.has(button?.name) ? "default" : "ghost"}
            size="sm"
            onClick={() => handleFormat(button?.command)}
            iconName={button?.icon}
            className="h-8 w-8 p-0"
          />
        ))}
        
        <div className="w-px h-6 bg-border mx-2" />
        
        <select
          className="px-2 py-1 text-sm bg-background border border-border rounded"
          onChange={(e) => {
            if (e?.target?.value) {
              document.execCommand('formatBlock', false, e?.target?.value);
              editorRef?.current?.focus();
            }
          }}
          defaultValue=""
        >
          <option value="">Formato</option>
          <option value="h1">Título 1</option>
          <option value="h2">Título 2</option>
          <option value="h3">Título 3</option>
          <option value="p">Parágrafo</option>
        </select>
      </div>
      {/* Editor */}
      <div
        ref={editorRef}
        contentEditable
        className={`min-h-[300px] p-4 bg-background border border-border rounded-b-lg focus:outline-none focus:ring-2 focus:ring-primary/20 ${
          error ? 'border-destructive' : ''
        }`}
        style={{
          lineHeight: '1.6',
          fontSize: '14px'
        }}
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onMouseUp={updateActiveFormats}
        onKeyUp={updateActiveFormats}
        dangerouslySetInnerHTML={{ __html: value || '' }}
        suppressContentEditableWarning={true}
      />
      {error && (
        <p className="text-sm text-destructive mt-1">{error}</p>
      )}
      <p className="text-xs text-muted-foreground">
        Use a barra de ferramentas para formatar o texto. Pressione Tab para adicionar indentação.
      </p>
    </div>
  );
};

export default RichTextEditor;