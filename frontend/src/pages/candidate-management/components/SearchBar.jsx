import React from 'react';
import Input from '../../../components/ui/Input';
import Icon from '../../../components/AppIcon';

const SearchBar = ({ searchTerm, onSearchChange, placeholder = "Buscar por nome, habilidades ou posição..." }) => {
  return (
    <div className="relative">
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Icon name="Search" size={20} className="text-muted-foreground" />
      </div>
      <Input
        type="search"
        placeholder={placeholder}
        value={searchTerm}
        onChange={(e) => onSearchChange(e?.target?.value)}
        className="pl-10 pr-4 py-2 w-full"
      />
      {searchTerm && (
        <button
          onClick={() => onSearchChange('')}
          className="absolute inset-y-0 right-0 pr-3 flex items-center text-muted-foreground hover:text-foreground transition-smooth"
        >
          <Icon name="X" size={16} />
        </button>
      )}
    </div>
  );
};

export default SearchBar;