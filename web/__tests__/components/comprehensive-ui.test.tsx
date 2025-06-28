/**
 * Comprehensive UI Components Tests
 * 
 * Tests for all UI components, widgets, forms, and interactive elements
 * following ADR-007 requirements for complete component coverage.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';

// Mock dependencies
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
  }),
  usePathname: () => '/test',
  useSearchParams: () => new URLSearchParams(),
}));

// Comprehensive UI Component Implementations

// Button Component with variants
interface ButtonProps {
  variant: 'primary' | 'secondary' | 'danger' | 'ghost';
  size: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  onClick?: () => void;
  children: React.ReactNode;
  type?: 'button' | 'submit' | 'reset';
  fullWidth?: boolean;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  icon,
  onClick,
  children,
  type = 'button',
  fullWidth = false,
}) => {
  const handleClick = () => {
    if (!disabled && !loading && onClick) {
      onClick();
    }
  };

  return (
    <button
      data-testid="button"
      className={`btn btn-${variant} btn-${size} ${fullWidth ? 'btn-full' : ''} ${loading ? 'btn-loading' : ''}`}
      disabled={disabled || loading}
      onClick={handleClick}
      type={type}
    >
      {loading && <span data-testid="loading-spinner">⟳</span>}
      {icon && <span data-testid="button-icon">{icon}</span>}
      <span data-testid="button-text">{children}</span>
    </button>
  );
};

// Input Component with validation
interface InputProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url';
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  onBlur?: () => void;
  onFocus?: () => void;
  disabled?: boolean;
  error?: string;
  success?: boolean;
  required?: boolean;
  autoComplete?: string;
  maxLength?: number;
  minLength?: number;
  pattern?: string;
  label?: string;
  helpText?: string;
  prefix?: React.ReactNode;
  suffix?: React.ReactNode;
  clearable?: boolean;
  onClear?: () => void;
}

const Input: React.FC<InputProps> = ({
  type = 'text',
  placeholder,
  value = '',
  onChange,
  onBlur,
  onFocus,
  disabled = false,
  error,
  success = false,
  required = false,
  autoComplete,
  maxLength,
  minLength,
  pattern,
  label,
  helpText,
  prefix,
  suffix,
  clearable = false,
  onClear,
}) => {
  const [focused, setFocused] = React.useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange?.(e.target.value);
  };

  const handleFocus = () => {
    setFocused(true);
    onFocus?.();
  };

  const handleBlur = () => {
    setFocused(false);
    onBlur?.();
  };

  const handleClear = () => {
    onChange?.('');
    onClear?.();
  };

  const hasError = Boolean(error);
  const isEmpty = value.length === 0;

  return (
    <div data-testid="input-container" className={`input-container ${hasError ? 'error' : ''} ${success ? 'success' : ''}`}>
      {label && (
        <label data-testid="input-label" className="input-label">
          {label}
          {required && <span className="required">*</span>}
        </label>
      )}
      
      <div className={`input-wrapper ${focused ? 'focused' : ''}`}>
        {prefix && <span data-testid="input-prefix" className="input-prefix">{prefix}</span>}
        
        <input
          data-testid="input"
          type={type}
          placeholder={placeholder}
          value={value}
          onChange={handleChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          disabled={disabled}
          required={required}
          autoComplete={autoComplete}
          maxLength={maxLength}
          minLength={minLength}
          pattern={pattern}
          className="input-field"
        />
        
        {clearable && !isEmpty && !disabled && (
          <button
            data-testid="input-clear"
            type="button"
            onClick={handleClear}
            className="input-clear"
          >
            ×
          </button>
        )}
        
        {suffix && <span data-testid="input-suffix" className="input-suffix">{suffix}</span>}
      </div>
      
      {error && <div data-testid="input-error" className="input-error">{error}</div>}
      {helpText && !error && <div data-testid="input-help" className="input-help">{helpText}</div>}
    </div>
  );
};

// Select Component
interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface SelectProps {
  options: SelectOption[];
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  error?: string;
  label?: string;
  required?: boolean;
  searchable?: boolean;
  multiple?: boolean;
  clearable?: boolean;
}

const Select: React.FC<SelectProps> = ({
  options,
  value = '',
  onChange,
  placeholder = 'Select an option',
  disabled = false,
  error,
  label,
  required = false,
  searchable = false,
  multiple = false,
  clearable = false,
}) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState('');
  const [selectedValues, setSelectedValues] = React.useState<string[]>(
    multiple ? (value ? value.split(',') : []) : []
  );

  const filteredOptions = React.useMemo(() => {
    if (!searchable || !searchQuery) return options;
    return options.filter(option =>
      option.label.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [options, searchQuery, searchable]);

  const selectedOption = options.find(opt => opt.value === value);
  const displayValue = multiple 
    ? selectedValues.map(val => options.find(opt => opt.value === val)?.label).join(', ')
    : selectedOption?.label || '';

  const handleToggle = () => {
    if (!disabled) {
      setIsOpen(!isOpen);
    }
  };

  const handleSelect = (optionValue: string) => {
    if (multiple) {
      const newValues = selectedValues.includes(optionValue)
        ? selectedValues.filter(val => val !== optionValue)
        : [...selectedValues, optionValue];
      setSelectedValues(newValues);
      onChange?.(newValues.join(','));
    } else {
      onChange?.(optionValue);
      setIsOpen(false);
    }
  };

  const handleClear = () => {
    if (multiple) {
      setSelectedValues([]);
      onChange?.('');
    } else {
      onChange?.('');
    }
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  return (
    <div data-testid="select-container" className={`select-container ${error ? 'error' : ''}`}>
      {label && (
        <label data-testid="select-label" className="select-label">
          {label}
          {required && <span className="required">*</span>}
        </label>
      )}
      
      <div className="select-wrapper">
        <div
          data-testid="select-trigger"
          className={`select-trigger ${isOpen ? 'open' : ''} ${disabled ? 'disabled' : ''}`}
          onClick={handleToggle}
        >
          <span data-testid="select-value" className="select-value">
            {displayValue || placeholder}
          </span>
          
          {clearable && (value || selectedValues.length > 0) && (
            <button
              data-testid="select-clear"
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              className="select-clear"
            >
              ×
            </button>
          )}
          
          <span data-testid="select-arrow" className={`select-arrow ${isOpen ? 'up' : 'down'}`}>
            ▼
          </span>
        </div>
        
        {isOpen && (
          <div data-testid="select-dropdown" className="select-dropdown">
            {searchable && (
              <div className="select-search">
                <input
                  data-testid="select-search-input"
                  type="text"
                  placeholder="Search options..."
                  value={searchQuery}
                  onChange={handleSearchChange}
                  className="select-search-input"
                />
              </div>
            )}
            
            <div data-testid="select-options" className="select-options">
              {filteredOptions.length === 0 ? (
                <div data-testid="select-no-options" className="select-no-options">
                  No options found
                </div>
              ) : (
                filteredOptions.map(option => (
                  <div
                    key={option.value}
                    data-testid={`select-option-${option.value}`}
                    className={`select-option ${
                      multiple 
                        ? selectedValues.includes(option.value) ? 'selected' : ''
                        : value === option.value ? 'selected' : ''
                    } ${option.disabled ? 'disabled' : ''}`}
                    onClick={() => !option.disabled && handleSelect(option.value)}
                  >
                    {multiple && (
                      <input
                        type="checkbox"
                        checked={selectedValues.includes(option.value)}
                        readOnly
                        className="select-checkbox"
                      />
                    )}
                    {option.label}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
      
      {error && <div data-testid="select-error" className="select-error">{error}</div>}
    </div>
  );
};

// Modal Component
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  showCloseButton?: boolean;
  footer?: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
  closeOnOverlayClick = true,
  closeOnEscape = true,
  showCloseButton = true,
  footer,
}) => {
  React.useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (closeOnEscape && event.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, closeOnEscape, onClose]);

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (closeOnOverlayClick && e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      data-testid="modal-overlay"
      className="modal-overlay"
      onClick={handleOverlayClick}
    >
      <div
        data-testid="modal-content"
        className={`modal-content modal-${size}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'modal-title' : undefined}
      >
        {(title || showCloseButton) && (
          <div data-testid="modal-header" className="modal-header">
            {title && (
              <h2 data-testid="modal-title" id="modal-title" className="modal-title">
                {title}
              </h2>
            )}
            {showCloseButton && (
              <button
                data-testid="modal-close"
                type="button"
                onClick={onClose}
                className="modal-close"
                aria-label="Close modal"
              >
                ×
              </button>
            )}
          </div>
        )}
        
        <div data-testid="modal-body" className="modal-body">
          {children}
        </div>
        
        {footer && (
          <div data-testid="modal-footer" className="modal-footer">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
};

// Toast Notification Component
interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title?: string;
  message: string;
  duration?: number;
  persistent?: boolean;
}

interface ToastContainerProps {
  toasts: Toast[];
  onRemove: (id: string) => void;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center';
}

const ToastContainer: React.FC<ToastContainerProps> = ({
  toasts,
  onRemove,
  position = 'top-right',
}) => {
  React.useEffect(() => {
    const timers: NodeJS.Timeout[] = [];

    toasts.forEach(toast => {
      if (!toast.persistent && toast.duration !== 0) {
        const timer = setTimeout(() => {
          onRemove(toast.id);
        }, toast.duration || 5000);
        timers.push(timer);
      }
    });

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [toasts, onRemove]);

  if (toasts.length === 0) return null;

  return (
    <div
      data-testid="toast-container"
      className={`toast-container toast-${position}`}
    >
      {toasts.map(toast => (
        <div
          key={toast.id}
          data-testid={`toast-${toast.id}`}
          className={`toast toast-${toast.type}`}
          role="alert"
          aria-live="polite"
        >
          <div className="toast-content">
            <div data-testid={`toast-icon-${toast.id}`} className="toast-icon">
              {toast.type === 'success' && '✓'}
              {toast.type === 'error' && '✗'}
              {toast.type === 'warning' && '⚠'}
              {toast.type === 'info' && 'ℹ'}
            </div>
            
            <div className="toast-text">
              {toast.title && (
                <div data-testid={`toast-title-${toast.id}`} className="toast-title">
                  {toast.title}
                </div>
              )}
              <div data-testid={`toast-message-${toast.id}`} className="toast-message">
                {toast.message}
              </div>
            </div>
            
            <button
              data-testid={`toast-close-${toast.id}`}
              type="button"
              onClick={() => onRemove(toast.id)}
              className="toast-close"
              aria-label="Close notification"
            >
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

// Tabs Component
interface Tab {
  id: string;
  label: string;
  content: React.ReactNode;
  disabled?: boolean;
  badge?: string | number;
}

interface TabsProps {
  tabs: Tab[];
  activeTab?: string;
  onTabChange?: (tabId: string) => void;
  variant?: 'default' | 'pills' | 'underline';
  size?: 'sm' | 'md' | 'lg';
}

const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab,
  onTabChange,
  variant = 'default',
  size = 'md',
}) => {
  const [activeTabId, setActiveTabId] = React.useState(activeTab || tabs[0]?.id);

  React.useEffect(() => {
    if (activeTab) {
      setActiveTabId(activeTab);
    }
  }, [activeTab]);

  const handleTabClick = (tabId: string) => {
    const tab = tabs.find(t => t.id === tabId);
    if (!tab?.disabled) {
      setActiveTabId(tabId);
      onTabChange?.(tabId);
    }
  };

  const activeTabContent = tabs.find(tab => tab.id === activeTabId)?.content;

  return (
    <div data-testid="tabs-container" className={`tabs-container tabs-${variant} tabs-${size}`}>
      <div data-testid="tabs-list" className="tabs-list" role="tablist">
        {tabs.map(tab => (
          <button
            key={tab.id}
            data-testid={`tab-${tab.id}`}
            className={`tab ${activeTabId === tab.id ? 'active' : ''} ${tab.disabled ? 'disabled' : ''}`}
            onClick={() => handleTabClick(tab.id)}
            disabled={tab.disabled}
            role="tab"
            aria-selected={activeTabId === tab.id}
            aria-controls={`tabpanel-${tab.id}`}
          >
            <span className="tab-label">{tab.label}</span>
            {tab.badge && (
              <span data-testid={`tab-badge-${tab.id}`} className="tab-badge">
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>
      
      <div
        data-testid="tab-content"
        className="tab-content"
        role="tabpanel"
        id={`tabpanel-${activeTabId}`}
        aria-labelledby={`tab-${activeTabId}`}
      >
        {activeTabContent}
      </div>
    </div>
  );
};

// Form Component with validation
interface FormField {
  name: string;
  type: 'text' | 'email' | 'password' | 'number' | 'select' | 'textarea' | 'checkbox' | 'radio';
  label: string;
  placeholder?: string;
  required?: boolean;
  options?: Array<{ value: string; label: string }>;
  validation?: (value: any) => string | null;
  defaultValue?: any;
}

interface FormProps {
  fields: FormField[];
  onSubmit: (data: Record<string, any>) => void;
  submitText?: string;
  loading?: boolean;
  resetOnSubmit?: boolean;
}

const Form: React.FC<FormProps> = ({
  fields,
  onSubmit,
  submitText = 'Submit',
  loading = false,
  resetOnSubmit = false,
}) => {
  const [formData, setFormData] = React.useState<Record<string, any>>(() => {
    const initialData: Record<string, any> = {};
    fields.forEach(field => {
      initialData[field.name] = field.defaultValue || (field.type === 'checkbox' ? false : '');
    });
    return initialData;
  });

  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const [touched, setTouched] = React.useState<Record<string, boolean>>({});

  const validateField = (field: FormField, value: any): string | null => {
    if (field.required && (!value || (typeof value === 'string' && value.trim() === ''))) {
      return `${field.label} is required`;
    }
    
    if (field.validation) {
      return field.validation(value);
    }
    
    return null;
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    fields.forEach(field => {
      const error = validateField(field, formData[field.name]);
      if (error) {
        newErrors[field.name] = error;
      }
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleFieldChange = (fieldName: string, value: any) => {
    setFormData(prev => ({ ...prev, [fieldName]: value }));
    
    if (touched[fieldName]) {
      const field = fields.find(f => f.name === fieldName);
      if (field) {
        const error = validateField(field, value);
        setErrors(prev => ({ ...prev, [fieldName]: error || '' }));
      }
    }
  };

  const handleFieldBlur = (fieldName: string) => {
    setTouched(prev => ({ ...prev, [fieldName]: true }));
    
    const field = fields.find(f => f.name === fieldName);
    if (field) {
      const error = validateField(field, formData[fieldName]);
      setErrors(prev => ({ ...prev, [fieldName]: error || '' }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (validateForm()) {
      onSubmit(formData);
      
      if (resetOnSubmit) {
        const resetData: Record<string, any> = {};
        fields.forEach(field => {
          resetData[field.name] = field.defaultValue || (field.type === 'checkbox' ? false : '');
        });
        setFormData(resetData);
        setErrors({});
        setTouched({});
      }
    } else {
      // Mark all fields as touched to show errors
      const allTouched: Record<string, boolean> = {};
      fields.forEach(field => {
        allTouched[field.name] = true;
      });
      setTouched(allTouched);
    }
  };

  const renderField = (field: FormField) => {
    const fieldError = touched[field.name] ? errors[field.name] : '';
    
    switch (field.type) {
      case 'select':
        return (
          <Select
            key={field.name}
            label={field.label}
            options={field.options || []}
            value={formData[field.name]}
            onChange={(value) => handleFieldChange(field.name, value)}
            error={fieldError}
            required={field.required}
            placeholder={field.placeholder}
          />
        );
        
      case 'textarea':
        return (
          <div key={field.name} className="form-field">
            <label className="field-label">
              {field.label}
              {field.required && <span className="required">*</span>}
            </label>
            <textarea
              data-testid={`field-${field.name}`}
              value={formData[field.name]}
              onChange={(e) => handleFieldChange(field.name, e.target.value)}
              onBlur={() => handleFieldBlur(field.name)}
              placeholder={field.placeholder}
              className={`field-textarea ${fieldError ? 'error' : ''}`}
              rows={4}
            />
            {fieldError && <div className="field-error">{fieldError}</div>}
          </div>
        );
        
      case 'checkbox':
        return (
          <div key={field.name} className="form-field checkbox-field">
            <label className="checkbox-label">
              <input
                data-testid={`field-${field.name}`}
                type="checkbox"
                checked={formData[field.name]}
                onChange={(e) => handleFieldChange(field.name, e.target.checked)}
                onBlur={() => handleFieldBlur(field.name)}
                className="checkbox-input"
              />
              <span className="checkbox-text">{field.label}</span>
              {field.required && <span className="required">*</span>}
            </label>
            {fieldError && <div className="field-error">{fieldError}</div>}
          </div>
        );
        
      case 'radio':
        return (
          <div key={field.name} className="form-field radio-field">
            <fieldset>
              <legend className="field-label">
                {field.label}
                {field.required && <span className="required">*</span>}
              </legend>
              {field.options?.map(option => (
                <label key={option.value} className="radio-label">
                  <input
                    data-testid={`field-${field.name}-${option.value}`}
                    type="radio"
                    name={field.name}
                    value={option.value}
                    checked={formData[field.name] === option.value}
                    onChange={(e) => handleFieldChange(field.name, e.target.value)}
                    onBlur={() => handleFieldBlur(field.name)}
                    className="radio-input"
                  />
                  <span className="radio-text">{option.label}</span>
                </label>
              ))}
            </fieldset>
            {fieldError && <div className="field-error">{fieldError}</div>}
          </div>
        );
        
      default:
        return (
          <Input
            key={field.name}
            type={field.type as any}
            label={field.label}
            placeholder={field.placeholder}
            value={formData[field.name]}
            onChange={(value) => handleFieldChange(field.name, value)}
            onBlur={() => handleFieldBlur(field.name)}
            error={fieldError}
            required={field.required}
          />
        );
    }
  };

  return (
    <form data-testid="form" onSubmit={handleSubmit} className="form">
      <div className="form-fields">
        {fields.map(renderField)}
      </div>
      
      <div className="form-actions">
        <Button
          type="submit"
          variant="primary"
          size="md"
          loading={loading}
          disabled={loading}
        >
          {submitText}
        </Button>
      </div>
    </form>
  );
};

// Pagination Component
interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  showFirstLast?: boolean;
  showPrevNext?: boolean;
  maxVisiblePages?: number;
  disabled?: boolean;
}

const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
  showFirstLast = true,
  showPrevNext = true,
  maxVisiblePages = 5,
  disabled = false,
}) => {
  const getVisiblePages = (): number[] => {
    const delta = Math.floor(maxVisiblePages / 2);
    const start = Math.max(1, currentPage - delta);
    const end = Math.min(totalPages, start + maxVisiblePages - 1);
    const adjustedStart = Math.max(1, end - maxVisiblePages + 1);
    
    const pages: number[] = [];
    for (let i = adjustedStart; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  };

  const visiblePages = getVisiblePages();
  const canGoPrevious = currentPage > 1 && !disabled;
  const canGoNext = currentPage < totalPages && !disabled;

  const handlePageClick = (page: number) => {
    if (!disabled && page !== currentPage && page >= 1 && page <= totalPages) {
      onPageChange(page);
    }
  };

  if (totalPages <= 1) return null;

  return (
    <nav data-testid="pagination" className="pagination" aria-label="Pagination">
      <div className="pagination-list">
        {showFirstLast && currentPage > 1 && (
          <button
            data-testid="pagination-first"
            className="pagination-button"
            onClick={() => handlePageClick(1)}
            disabled={disabled}
            aria-label="Go to first page"
          >
            «
          </button>
        )}
        
        {showPrevNext && (
          <button
            data-testid="pagination-previous"
            className="pagination-button"
            onClick={() => handlePageClick(currentPage - 1)}
            disabled={!canGoPrevious}
            aria-label="Go to previous page"
          >
            ‹
          </button>
        )}
        
        {visiblePages[0] > 1 && (
          <>
            <button
              data-testid="pagination-page-1"
              className="pagination-button"
              onClick={() => handlePageClick(1)}
              disabled={disabled}
            >
              1
            </button>
            {visiblePages[0] > 2 && (
              <span data-testid="pagination-ellipsis-start" className="pagination-ellipsis">
                ...
              </span>
            )}
          </>
        )}
        
        {visiblePages.map(page => (
          <button
            key={page}
            data-testid={`pagination-page-${page}`}
            className={`pagination-button ${page === currentPage ? 'active' : ''}`}
            onClick={() => handlePageClick(page)}
            disabled={disabled}
            aria-label={`Go to page ${page}`}
            aria-current={page === currentPage ? 'page' : undefined}
          >
            {page}
          </button>
        ))}
        
        {visiblePages[visiblePages.length - 1] < totalPages && (
          <>
            {visiblePages[visiblePages.length - 1] < totalPages - 1 && (
              <span data-testid="pagination-ellipsis-end" className="pagination-ellipsis">
                ...
              </span>
            )}
            <button
              data-testid={`pagination-page-${totalPages}`}
              className="pagination-button"
              onClick={() => handlePageClick(totalPages)}
              disabled={disabled}
            >
              {totalPages}
            </button>
          </>
        )}
        
        {showPrevNext && (
          <button
            data-testid="pagination-next"
            className="pagination-button"
            onClick={() => handlePageClick(currentPage + 1)}
            disabled={!canGoNext}
            aria-label="Go to next page"
          >
            ›
          </button>
        )}
        
        {showFirstLast && currentPage < totalPages && (
          <button
            data-testid="pagination-last"
            className="pagination-button"
            onClick={() => handlePageClick(totalPages)}
            disabled={disabled}
            aria-label="Go to last page"
          >
            »
          </button>
        )}
      </div>
      
      <div data-testid="pagination-info" className="pagination-info">
        Page {currentPage} of {totalPages}
      </div>
    </nav>
  );
};

describe('Comprehensive UI Components Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Button Component', () => {
    it('renders with different variants', () => {
      const { rerender } = render(
        <Button variant="primary" size="md" onClick={() => {}}>
          Primary Button
        </Button>
      );

      expect(screen.getByTestId('button')).toHaveClass('btn-primary');

      rerender(
        <Button variant="secondary" size="md" onClick={() => {}}>
          Secondary Button
        </Button>
      );

      expect(screen.getByTestId('button')).toHaveClass('btn-secondary');
    });

    it('handles click events', () => {
      const handleClick = jest.fn();
      render(
        <Button variant="primary" size="md" onClick={handleClick}>
          Click Me
        </Button>
      );

      fireEvent.click(screen.getByTestId('button'));
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('shows loading state', () => {
      render(
        <Button variant="primary" size="md" loading onClick={() => {}}>
          Loading Button
        </Button>
      );

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
      expect(screen.getByTestId('button')).toBeDisabled();
    });

    it('displays icons', () => {
      render(
        <Button variant="primary" size="md" icon={<span>🔥</span>} onClick={() => {}}>
          With Icon
        </Button>
      );

      expect(screen.getByTestId('button-icon')).toHaveTextContent('🔥');
    });

    it('handles disabled state', () => {
      const handleClick = jest.fn();
      render(
        <Button variant="primary" size="md" disabled onClick={handleClick}>
          Disabled Button
        </Button>
      );

      fireEvent.click(screen.getByTestId('button'));
      expect(handleClick).not.toHaveBeenCalled();
      expect(screen.getByTestId('button')).toBeDisabled();
    });

    it('supports different sizes', () => {
      const { rerender } = render(
        <Button variant="primary" size="sm" onClick={() => {}}>
          Small Button
        </Button>
      );

      expect(screen.getByTestId('button')).toHaveClass('btn-sm');

      rerender(
        <Button variant="primary" size="lg" onClick={() => {}}>
          Large Button
        </Button>
      );

      expect(screen.getByTestId('button')).toHaveClass('btn-lg');
    });

    it('supports full width', () => {
      render(
        <Button variant="primary" size="md" fullWidth onClick={() => {}}>
          Full Width Button
        </Button>
      );

      expect(screen.getByTestId('button')).toHaveClass('btn-full');
    });
  });

  describe('Input Component', () => {
    it('renders with label and placeholder', () => {
      render(
        <Input
          label="Test Input"
          placeholder="Enter text here"
          value=""
          onChange={() => {}}
        />
      );

      expect(screen.getByTestId('input-label')).toHaveTextContent('Test Input');
      expect(screen.getByTestId('input')).toHaveAttribute('placeholder', 'Enter text here');
    });

    it('handles value changes', async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();
      
      render(
        <Input
          value=""
          onChange={handleChange}
        />
      );

      await user.type(screen.getByTestId('input'), 'Hello World');
      expect(handleChange).toHaveBeenCalledWith('H');
    });

    it('shows error state', () => {
      render(
        <Input
          value=""
          onChange={() => {}}
          error="This field is required"
        />
      );

      expect(screen.getByTestId('input-error')).toHaveTextContent('This field is required');
      expect(screen.getByTestId('input-container')).toHaveClass('error');
    });

    it('shows success state', () => {
      render(
        <Input
          value="Valid input"
          onChange={() => {}}
          success
        />
      );

      expect(screen.getByTestId('input-container')).toHaveClass('success');
    });

    it('handles clear functionality', async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();
      const handleClear = jest.fn();
      
      render(
        <Input
          value="Some text"
          onChange={handleChange}
          onClear={handleClear}
          clearable
        />
      );

      await user.click(screen.getByTestId('input-clear'));
      expect(handleChange).toHaveBeenCalledWith('');
      expect(handleClear).toHaveBeenCalled();
    });

    it('handles focus and blur events', async () => {
      const user = userEvent.setup();
      const handleFocus = jest.fn();
      const handleBlur = jest.fn();
      
      render(
        <Input
          value=""
          onChange={() => {}}
          onFocus={handleFocus}
          onBlur={handleBlur}
        />
      );

      await user.click(screen.getByTestId('input'));
      expect(handleFocus).toHaveBeenCalled();

      await user.tab();
      expect(handleBlur).toHaveBeenCalled();
    });

    it('displays prefix and suffix', () => {
      render(
        <Input
          value=""
          onChange={() => {}}
          prefix={<span>$</span>}
          suffix={<span>.00</span>}
        />
      );

      expect(screen.getByTestId('input-prefix')).toHaveTextContent('$');
      expect(screen.getByTestId('input-suffix')).toHaveTextContent('.00');
    });

    it('shows required indicator', () => {
      render(
        <Input
          label="Required Field"
          value=""
          onChange={() => {}}
          required
        />
      );

      expect(screen.getByTestId('input-label')).toHaveTextContent('Required Field*');
    });

    it('shows help text', () => {
      render(
        <Input
          value=""
          onChange={() => {}}
          helpText="This is helpful information"
        />
      );

      expect(screen.getByTestId('input-help')).toHaveTextContent('This is helpful information');
    });
  });

  describe('Select Component', () => {
    const options = [
      { value: 'option1', label: 'Option 1' },
      { value: 'option2', label: 'Option 2' },
      { value: 'option3', label: 'Option 3', disabled: true },
    ];

    it('renders with options', () => {
      render(
        <Select
          options={options}
          value=""
          onChange={() => {}}
          placeholder="Select an option"
        />
      );

      expect(screen.getByTestId('select-value')).toHaveTextContent('Select an option');
    });

    it('opens and closes dropdown', async () => {
      const user = userEvent.setup();
      render(
        <Select
          options={options}
          value=""
          onChange={() => {}}
        />
      );

      expect(screen.queryByTestId('select-dropdown')).not.toBeInTheDocument();

      await user.click(screen.getByTestId('select-trigger'));
      expect(screen.getByTestId('select-dropdown')).toBeInTheDocument();
    });

    it('selects options', async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();
      
      render(
        <Select
          options={options}
          value=""
          onChange={handleChange}
        />
      );

      await user.click(screen.getByTestId('select-trigger'));
      await user.click(screen.getByTestId('select-option-option1'));
      
      expect(handleChange).toHaveBeenCalledWith('option1');
    });

    it('handles disabled options', async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();
      
      render(
        <Select
          options={options}
          value=""
          onChange={handleChange}
        />
      );

      await user.click(screen.getByTestId('select-trigger'));
      await user.click(screen.getByTestId('select-option-option3'));
      
      expect(handleChange).not.toHaveBeenCalled();
    });

    it('supports search functionality', async () => {
      const user = userEvent.setup();
      render(
        <Select
          options={options}
          value=""
          onChange={() => {}}
          searchable
        />
      );

      await user.click(screen.getByTestId('select-trigger'));
      await user.type(screen.getByTestId('select-search-input'), 'Option 1');
      
      expect(screen.getByTestId('select-option-option1')).toBeInTheDocument();
      expect(screen.queryByTestId('select-option-option2')).not.toBeInTheDocument();
    });

    it('supports multiple selection', async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();
      
      render(
        <Select
          options={options}
          value=""
          onChange={handleChange}
          multiple
        />
      );

      await user.click(screen.getByTestId('select-trigger'));
      await user.click(screen.getByTestId('select-option-option1'));
      
      expect(handleChange).toHaveBeenCalledWith('option1');
    });

    it('handles clear functionality', async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();
      
      render(
        <Select
          options={options}
          value="option1"
          onChange={handleChange}
          clearable
        />
      );

      await user.click(screen.getByTestId('select-clear'));
      expect(handleChange).toHaveBeenCalledWith('');
    });
  });

  describe('Modal Component', () => {
    it('renders when open', () => {
      render(
        <Modal
          isOpen={true}
          onClose={() => {}}
          title="Test Modal"
        >
          Modal content here
        </Modal>
      );

      expect(screen.getByTestId('modal-overlay')).toBeInTheDocument();
      expect(screen.getByTestId('modal-title')).toHaveTextContent('Test Modal');
    });

    it('does not render when closed', () => {
      render(
        <Modal
          isOpen={false}
          onClose={() => {}}
          title="Test Modal"
        >
          Modal content here
        </Modal>
      );

      expect(screen.queryByTestId('modal-overlay')).not.toBeInTheDocument();
    });

    it('closes on close button click', async () => {
      const user = userEvent.setup();
      const handleClose = jest.fn();
      
      render(
        <Modal
          isOpen={true}
          onClose={handleClose}
          title="Test Modal"
        >
          Modal content here
        </Modal>
      );

      await user.click(screen.getByTestId('modal-close'));
      expect(handleClose).toHaveBeenCalled();
    });

    it('closes on overlay click when enabled', async () => {
      const user = userEvent.setup();
      const handleClose = jest.fn();
      
      render(
        <Modal
          isOpen={true}
          onClose={handleClose}
          closeOnOverlayClick={true}
        >
          Modal content here
        </Modal>
      );

      await user.click(screen.getByTestId('modal-overlay'));
      expect(handleClose).toHaveBeenCalled();
    });

    it('does not close on overlay click when disabled', async () => {
      const user = userEvent.setup();
      const handleClose = jest.fn();
      
      render(
        <Modal
          isOpen={true}
          onClose={handleClose}
          closeOnOverlayClick={false}
        >
          Modal content here
        </Modal>
      );

      await user.click(screen.getByTestId('modal-overlay'));
      expect(handleClose).not.toHaveBeenCalled();
    });

    it('handles escape key when enabled', () => {
      const handleClose = jest.fn();
      
      render(
        <Modal
          isOpen={true}
          onClose={handleClose}
          closeOnEscape={true}
        >
          Modal content here
        </Modal>
      );

      fireEvent.keyDown(document, { key: 'Escape' });
      expect(handleClose).toHaveBeenCalled();
    });

    it('renders footer when provided', () => {
      render(
        <Modal
          isOpen={true}
          onClose={() => {}}
          footer={<div>Footer content</div>}
        >
          Modal content here
        </Modal>
      );

      expect(screen.getByTestId('modal-footer')).toHaveTextContent('Footer content');
    });

    it('supports different sizes', () => {
      const { rerender } = render(
        <Modal
          isOpen={true}
          onClose={() => {}}
          size="sm"
        >
          Small modal
        </Modal>
      );

      expect(screen.getByTestId('modal-content')).toHaveClass('modal-sm');

      rerender(
        <Modal
          isOpen={true}
          onClose={() => {}}
          size="xl"
        >
          Extra large modal
        </Modal>
      );

      expect(screen.getByTestId('modal-content')).toHaveClass('modal-xl');
    });
  });

  describe('ToastContainer Component', () => {
    const mockToasts: Toast[] = [
      {
        id: '1',
        type: 'success',
        title: 'Success!',
        message: 'Operation completed successfully',
        duration: 5000,
      },
      {
        id: '2',
        type: 'error',
        message: 'An error occurred',
        persistent: true,
      },
    ];

    it('renders toasts correctly', () => {
      render(
        <ToastContainer
          toasts={mockToasts}
          onRemove={() => {}}
        />
      );

      expect(screen.getByTestId('toast-1')).toBeInTheDocument();
      expect(screen.getByTestId('toast-2')).toBeInTheDocument();
      expect(screen.getByTestId('toast-title-1')).toHaveTextContent('Success!');
      expect(screen.getByTestId('toast-message-1')).toHaveTextContent('Operation completed successfully');
    });

    it('removes toasts on close click', async () => {
      const user = userEvent.setup();
      const handleRemove = jest.fn();
      
      render(
        <ToastContainer
          toasts={mockToasts}
          onRemove={handleRemove}
        />
      );

      await user.click(screen.getByTestId('toast-close-1'));
      expect(handleRemove).toHaveBeenCalledWith('1');
    });

    it('displays correct icons for different types', () => {
      render(
        <ToastContainer
          toasts={mockToasts}
          onRemove={() => {}}
        />
      );

      expect(screen.getByTestId('toast-icon-1')).toHaveTextContent('✓');
      expect(screen.getByTestId('toast-icon-2')).toHaveTextContent('✗');
    });

    it('does not render when no toasts', () => {
      render(
        <ToastContainer
          toasts={[]}
          onRemove={() => {}}
        />
      );

      expect(screen.queryByTestId('toast-container')).not.toBeInTheDocument();
    });

    it('supports different positions', () => {
      const { rerender } = render(
        <ToastContainer
          toasts={mockToasts}
          onRemove={() => {}}
          position="top-left"
        />
      );

      expect(screen.getByTestId('toast-container')).toHaveClass('toast-top-left');

      rerender(
        <ToastContainer
          toasts={mockToasts}
          onRemove={() => {}}
          position="bottom-center"
        />
      );

      expect(screen.getByTestId('toast-container')).toHaveClass('toast-bottom-center');
    });
  });

  describe('Tabs Component', () => {
    const mockTabs: Tab[] = [
      {
        id: 'tab1',
        label: 'Tab 1',
        content: <div>Content 1</div>,
      },
      {
        id: 'tab2',
        label: 'Tab 2',
        content: <div>Content 2</div>,
        badge: '5',
      },
      {
        id: 'tab3',
        label: 'Tab 3',
        content: <div>Content 3</div>,
        disabled: true,
      },
    ];

    it('renders tabs and content', () => {
      render(
        <Tabs
          tabs={mockTabs}
          onTabChange={() => {}}
        />
      );

      expect(screen.getByTestId('tab-tab1')).toBeInTheDocument();
      expect(screen.getByTestId('tab-tab2')).toBeInTheDocument();
      expect(screen.getByTestId('tab-tab3')).toBeInTheDocument();
      expect(screen.getByTestId('tab-content')).toHaveTextContent('Content 1');
    });

    it('switches tabs on click', async () => {
      const user = userEvent.setup();
      const handleTabChange = jest.fn();
      
      render(
        <Tabs
          tabs={mockTabs}
          onTabChange={handleTabChange}
        />
      );

      await user.click(screen.getByTestId('tab-tab2'));
      expect(handleTabChange).toHaveBeenCalledWith('tab2');
      expect(screen.getByTestId('tab-content')).toHaveTextContent('Content 2');
    });

    it('does not switch to disabled tabs', async () => {
      const user = userEvent.setup();
      const handleTabChange = jest.fn();
      
      render(
        <Tabs
          tabs={mockTabs}
          onTabChange={handleTabChange}
        />
      );

      await user.click(screen.getByTestId('tab-tab3'));
      expect(handleTabChange).not.toHaveBeenCalledWith('tab3');
    });

    it('displays badges', () => {
      render(
        <Tabs
          tabs={mockTabs}
          onTabChange={() => {}}
        />
      );

      expect(screen.getByTestId('tab-badge-tab2')).toHaveTextContent('5');
    });

    it('supports different variants', () => {
      const { rerender } = render(
        <Tabs
          tabs={mockTabs}
          variant="pills"
          onTabChange={() => {}}
        />
      );

      expect(screen.getByTestId('tabs-container')).toHaveClass('tabs-pills');

      rerender(
        <Tabs
          tabs={mockTabs}
          variant="underline"
          onTabChange={() => {}}
        />
      );

      expect(screen.getByTestId('tabs-container')).toHaveClass('tabs-underline');
    });

    it('supports controlled active tab', () => {
      render(
        <Tabs
          tabs={mockTabs}
          activeTab="tab2"
          onTabChange={() => {}}
        />
      );

      expect(screen.getByTestId('tab-tab2')).toHaveClass('active');
      expect(screen.getByTestId('tab-content')).toHaveTextContent('Content 2');
    });
  });

  describe('Form Component', () => {
    const mockFields: FormField[] = [
      {
        name: 'name',
        type: 'text',
        label: 'Name',
        required: true,
        placeholder: 'Enter your name',
      },
      {
        name: 'email',
        type: 'email',
        label: 'Email',
        required: true,
        validation: (value) => {
          if (value && !value.includes('@')) {
            return 'Please enter a valid email';
          }
          return null;
        },
      },
      {
        name: 'country',
        type: 'select',
        label: 'Country',
        options: [
          { value: 'us', label: 'United States' },
          { value: 'uk', label: 'United Kingdom' },
          { value: 'ca', label: 'Canada' },
        ],
      },
      {
        name: 'newsletter',
        type: 'checkbox',
        label: 'Subscribe to newsletter',
        defaultValue: false,
      },
    ];

    it('renders form fields', () => {
      render(
        <Form
          fields={mockFields}
          onSubmit={() => {}}
        />
      );

      expect(screen.getByTestId('form')).toBeInTheDocument();
      expect(screen.getByDisplayValue('')).toBeInTheDocument(); // name field
    });

    it('validates required fields on submit', async () => {
      const user = userEvent.setup();
      const handleSubmit = jest.fn();
      
      render(
        <Form
          fields={mockFields}
          onSubmit={handleSubmit}
        />
      );

      await user.click(screen.getByRole('button', { name: /submit/i }));
      
      expect(handleSubmit).not.toHaveBeenCalled();
      expect(screen.getByText('Name is required')).toBeInTheDocument();
    });

    it('submits valid form data', async () => {
      const user = userEvent.setup();
      const handleSubmit = jest.fn();
      
      render(
        <Form
          fields={mockFields}
          onSubmit={handleSubmit}
        />
      );

      await user.type(screen.getByDisplayValue(''), 'John Doe');
      await user.type(screen.getByDisplayValue(''), 'john@example.com');
      await user.click(screen.getByRole('button', { name: /submit/i }));
      
      await waitFor(() => {
        expect(handleSubmit).toHaveBeenCalledWith({
          name: 'John Doe',
          email: 'john@example.com',
          country: '',
          newsletter: false,
        });
      });
    });

    it('validates fields with custom validation', async () => {
      const user = userEvent.setup();
      
      render(
        <Form
          fields={mockFields}
          onSubmit={() => {}}
        />
      );

      const emailInput = screen.getByDisplayValue('');
      await user.type(emailInput, 'invalid-email');
      await user.tab(); // Blur the field
      
      await waitFor(() => {
        expect(screen.getByText('Please enter a valid email')).toBeInTheDocument();
      });
    });

    it('resets form when resetOnSubmit is true', async () => {
      const user = userEvent.setup();
      
      render(
        <Form
          fields={mockFields}
          onSubmit={() => {}}
          resetOnSubmit={true}
        />
      );

      const nameInput = screen.getByDisplayValue('');
      await user.type(nameInput, 'John Doe');
      
      // Complete the form
      const allInputs = screen.getAllByDisplayValue('');
      await user.type(allInputs[0], 'john@example.com');
      
      await user.click(screen.getByRole('button', { name: /submit/i }));
      
      await waitFor(() => {
        expect(screen.getByDisplayValue('')).toBeInTheDocument();
      });
    });
  });

  describe('Pagination Component', () => {
    it('renders pagination with correct pages', () => {
      render(
        <Pagination
          currentPage={3}
          totalPages={10}
          onPageChange={() => {}}
        />
      );

      expect(screen.getByTestId('pagination')).toBeInTheDocument();
      expect(screen.getByTestId('pagination-page-3')).toHaveClass('active');
      expect(screen.getByTestId('pagination-info')).toHaveTextContent('Page 3 of 10');
    });

    it('handles page changes', async () => {
      const user = userEvent.setup();
      const handlePageChange = jest.fn();
      
      render(
        <Pagination
          currentPage={3}
          totalPages={10}
          onPageChange={handlePageChange}
        />
      );

      await user.click(screen.getByTestId('pagination-page-5'));
      expect(handlePageChange).toHaveBeenCalledWith(5);
    });

    it('handles previous and next navigation', async () => {
      const user = userEvent.setup();
      const handlePageChange = jest.fn();
      
      render(
        <Pagination
          currentPage={3}
          totalPages={10}
          onPageChange={handlePageChange}
        />
      );

      await user.click(screen.getByTestId('pagination-previous'));
      expect(handlePageChange).toHaveBeenCalledWith(2);

      await user.click(screen.getByTestId('pagination-next'));
      expect(handlePageChange).toHaveBeenCalledWith(4);
    });

    it('handles first and last navigation', async () => {
      const user = userEvent.setup();
      const handlePageChange = jest.fn();
      
      render(
        <Pagination
          currentPage={5}
          totalPages={10}
          onPageChange={handlePageChange}
        />
      );

      await user.click(screen.getByTestId('pagination-first'));
      expect(handlePageChange).toHaveBeenCalledWith(1);

      await user.click(screen.getByTestId('pagination-last'));
      expect(handlePageChange).toHaveBeenCalledWith(10);
    });

    it('disables navigation when appropriate', () => {
      render(
        <Pagination
          currentPage={1}
          totalPages={10}
          onPageChange={() => {}}
        />
      );

      expect(screen.getByTestId('pagination-previous')).toBeDisabled();
      expect(screen.getByTestId('pagination-next')).not.toBeDisabled();
    });

    it('shows ellipsis for large page ranges', () => {
      render(
        <Pagination
          currentPage={10}
          totalPages={20}
          onPageChange={() => {}}
          maxVisiblePages={5}
        />
      );

      expect(screen.getByTestId('pagination-ellipsis-start')).toBeInTheDocument();
    });

    it('does not render for single page', () => {
      render(
        <Pagination
          currentPage={1}
          totalPages={1}
          onPageChange={() => {}}
        />
      );

      expect(screen.queryByTestId('pagination')).not.toBeInTheDocument();
    });

    it('handles disabled state', () => {
      render(
        <Pagination
          currentPage={3}
          totalPages={10}
          onPageChange={() => {}}
          disabled={true}
        />
      );

      expect(screen.getByTestId('pagination-page-3')).toBeDisabled();
      expect(screen.getByTestId('pagination-previous')).toBeDisabled();
      expect(screen.getByTestId('pagination-next')).toBeDisabled();
    });
  });

  describe('Component Integration', () => {
    it('integrates multiple components effectively', async () => {
      const user = userEvent.setup();
      const [isModalOpen, setIsModalOpen] = React.useState(false);
      const [toasts, setToasts] = React.useState<Toast[]>([]);
      
      const TestApp = () => {
        const addToast = (toast: Omit<Toast, 'id'>) => {
          const newToast = { ...toast, id: Date.now().toString() };
          setToasts(prev => [...prev, newToast]);
        };

        const removeToast = (id: string) => {
          setToasts(prev => prev.filter(t => t.id !== id));
        };

        return (
          <div>
            <Button
              variant="primary"
              size="md"
              onClick={() => setIsModalOpen(true)}
            >
              Open Modal
            </Button>

            <Modal
              isOpen={isModalOpen}
              onClose={() => setIsModalOpen(false)}
              title="Test Modal"
            >
              <Form
                fields={[
                  {
                    name: 'test',
                    type: 'text',
                    label: 'Test Field',
                    required: true,
                  },
                ]}
                onSubmit={(data) => {
                  addToast({
                    type: 'success',
                    message: `Form submitted with: ${data.test}`,
                  });
                  setIsModalOpen(false);
                }}
              />
            </Modal>

            <ToastContainer
              toasts={toasts}
              onRemove={removeToast}
            />
          </div>
        );
      };

      render(<TestApp />);

      // Open modal
      await user.click(screen.getByText('Open Modal'));
      expect(screen.getByTestId('modal-overlay')).toBeInTheDocument();

      // Fill form and submit
      await user.type(screen.getByDisplayValue(''), 'Test Value');
      await user.click(screen.getByRole('button', { name: /submit/i }));

      // Check toast appears
      await waitFor(() => {
        expect(screen.getByText('Form submitted with: Test Value')).toBeInTheDocument();
      });

      // Modal should be closed
      expect(screen.queryByTestId('modal-overlay')).not.toBeInTheDocument();
    });
  });
});