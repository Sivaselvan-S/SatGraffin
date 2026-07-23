import React, { useRef } from 'react';
import { Paperclip, X, Image as ImageIcon, FileText } from 'lucide-react';

interface FileUploadProps {
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  disabled?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({ selectedFile, onFileSelect, disabled }) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  const clear = () => {
    onFileSelect(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="file-upload">
      <input
        ref={inputRef}
        type="file"
        accept="image/*,application/pdf"
        className="file-upload__input"
        id="file-input-chat"
        onChange={handleChange}
        disabled={disabled}
        aria-label="Attach image or PDF"
      />

      {selectedFile ? (
        <div className="file-upload__preview">
          {selectedFile.type.startsWith('image/') ? (
            <ImageIcon size={14} aria-hidden />
          ) : (
            <FileText size={14} aria-hidden />
          )}
          <span className="file-upload__name">{selectedFile.name}</span>
          <button
            type="button"
            className="file-upload__clear"
            onClick={clear}
            disabled={disabled}
            aria-label="Remove attachment"
          >
            <X size={12} />
          </button>
        </div>
      ) : (
        <label
          htmlFor="file-input-chat"
          className={`file-upload__label${disabled ? ' file-upload__label--disabled' : ''}`}
          title="Upload image or PDF document"
        >
          <Paperclip size={15} aria-hidden />
          <span>Attach</span>
        </label>
      )}
    </div>
  );
};
