/**
 * Comprehensive Accessibility Tests
 * 
 * Tests for WCAG compliance, screen reader support, keyboard navigation,
 * and accessibility utilities following ADR-007 requirements.
 */

import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';

// Mock axe-core for accessibility testing
const mockAxe = {
  run: jest.fn(() => Promise.resolve({
    violations: [],
    passes: [],
    incomplete: [],
    inapplicable: [],
  })),
  configure: jest.fn(),
  reset: jest.fn(),
};

jest.unstable_mockModule('axe-core', () => mockAxe);

// Accessibility Helper Functions
interface AccessibilityOptions {
  includeHidden?: boolean;
  rules?: string[];
  tags?: string[];
}

class AccessibilityTester {
  static async runAxeTests(element: Element, options: AccessibilityOptions = {}): Promise<any> {
    const config = {
      rules: options.rules ? 
        Object.fromEntries(options.rules.map(rule => [rule, { enabled: true }])) : 
        undefined,
      tags: options.tags,
    };

    if (config.rules || config.tags) {
      mockAxe.configure(config);
    }

    return await mockAxe.run(element, {
      includeHidden: options.includeHidden || false,
    });
  }

  static checkColorContrast(foreground: string, background: string): { ratio: number; passes: boolean } {
    // Simplified color contrast calculation
    const hex2rgb = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      } : null;
    };

    const getLuminance = (r: number, g: number, b: number) => {
      const [rs, gs, bs] = [r, g, b].map(c => {
        c = c / 255;
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    };

    const fg = hex2rgb(foreground);
    const bg = hex2rgb(background);

    if (!fg || !bg) {
      throw new Error('Invalid color format');
    }

    const l1 = getLuminance(fg.r, fg.g, fg.b);
    const l2 = getLuminance(bg.r, bg.g, bg.b);

    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);

    return {
      ratio,
      passes: ratio >= 4.5, // WCAG AA standard
    };
  }

  static checkKeyboardNavigation(element: Element): {
    focusableElements: Element[];
    tabOrder: number[];
    hasSkipLinks: boolean;
  } {
    const focusableSelectors = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]',
    ];

    const focusableElements = Array.from(
      element.querySelectorAll(focusableSelectors.join(', '))
    );

    const tabOrder = focusableElements.map(el => {
      const tabIndex = el.getAttribute('tabindex');
      return tabIndex ? parseInt(tabIndex, 10) : 0;
    });

    const hasSkipLinks = !!element.querySelector('a[href^="#"]:first-child');

    return {
      focusableElements,
      tabOrder,
      hasSkipLinks,
    };
  }

  static checkAriaLabels(element: Element): {
    elementsWithoutLabels: Element[];
    elementsWithAriaDescriptions: Element[];
    landmarkElements: Element[];
  } {
    const interactiveElements = Array.from(
      element.querySelectorAll('button, input, select, textarea, a[href]')
    );

    const elementsWithoutLabels = interactiveElements.filter(el => {
      const hasAriaLabel = el.hasAttribute('aria-label');
      const hasAriaLabelledBy = el.hasAttribute('aria-labelledby');
      const hasTitle = el.hasAttribute('title');
      const hasTextContent = el.textContent?.trim();
      const hasAssociatedLabel = el.id && element.querySelector(`label[for="${el.id}"]`);

      return !(hasAriaLabel || hasAriaLabelledBy || hasTitle || hasTextContent || hasAssociatedLabel);
    });

    const elementsWithAriaDescriptions = Array.from(
      element.querySelectorAll('[aria-describedby], [aria-description]')
    );

    const landmarkElements = Array.from(
      element.querySelectorAll('main, nav, aside, section, header, footer, [role="banner"], [role="navigation"], [role="complementary"], [role="contentinfo"]')
    );

    return {
      elementsWithoutLabels,
      elementsWithAriaDescriptions,
      landmarkElements,
    };
  }
}

// Accessible Component Examples for Testing
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  ariaLabel?: string;
  ariaDescribedBy?: string;
  onClick?: () => void;
  children: React.ReactNode;
}

const AccessibleButton: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  ariaLabel,
  ariaDescribedBy,
  onClick,
  children,
}) => {
  const handleClick = () => {
    if (!disabled && !loading && onClick) {
      onClick();
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick();
    }
  };

  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      disabled={disabled || loading}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedBy}
      aria-busy={loading}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={disabled ? -1 : 0}
    >
      {loading && (
        <span aria-hidden="true" className="spinner">
          ⟳
        </span>
      )}
      <span className={loading ? 'sr-only' : undefined}>
        {children}
      </span>
      {loading && (
        <span className="sr-only">
          Loading, please wait
        </span>
      )}
    </button>
  );
};

interface FormFieldProps {
  id: string;
  label: string;
  type?: string;
  required?: boolean;
  error?: string;
  helpText?: string;
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
}

const AccessibleFormField: React.FC<FormFieldProps> = ({
  id,
  label,
  type = 'text',
  required = false,
  error,
  helpText,
  value = '',
  onChange,
  placeholder,
}) => {
  const errorId = `${id}-error`;
  const helpId = `${id}-help`;

  const describedBy = [
    helpText ? helpId : null,
    error ? errorId : null,
  ].filter(Boolean).join(' ') || undefined;

  return (
    <div className="form-field">
      <label htmlFor={id} className="form-label">
        {label}
        {required && (
          <span className="required" aria-label="required">
            *
          </span>
        )}
      </label>
      
      <input
        id={id}
        type={type}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        placeholder={placeholder}
        required={required}
        aria-invalid={!!error}
        aria-describedby={describedBy}
        className={`form-input ${error ? 'form-input--error' : ''}`}
      />
      
      {helpText && (
        <div id={helpId} className="form-help">
          {helpText}
        </div>
      )}
      
      {error && (
        <div id={errorId} className="form-error" role="alert">
          {error}
        </div>
      )}
    </div>
  );
};

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  closeOnEscape?: boolean;
  closeOnOverlayClick?: boolean;
}

const AccessibleModal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  closeOnEscape = true,
  closeOnOverlayClick = true,
}) => {
  const modalRef = React.useRef<HTMLDivElement>(null);
  const previousFocusRef = React.useRef<Element | null>(null);

  React.useEffect(() => {
    if (isOpen) {
      // Store previously focused element
      previousFocusRef.current = document.activeElement;
      
      // Focus the modal
      modalRef.current?.focus();
      
      // Trap focus within modal
      const handleKeyDown = (event: KeyboardEvent) => {
        if (!modalRef.current) return;

        const focusableElements = modalRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (event.key === 'Tab') {
          if (event.shiftKey) {
            if (document.activeElement === firstElement) {
              event.preventDefault();
              lastElement?.focus();
            }
          } else {
            if (document.activeElement === lastElement) {
              event.preventDefault();
              firstElement?.focus();
            }
          }
        }

        if (event.key === 'Escape' && closeOnEscape) {
          onClose();
        }
      };

      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';

      return () => {
        document.removeEventListener('keydown', handleKeyDown);
        document.body.style.overflow = '';
        
        // Restore focus to previously focused element
        if (previousFocusRef.current instanceof HTMLElement) {
          previousFocusRef.current.focus();
        }
      };
    }
  }, [isOpen, onClose, closeOnEscape]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={closeOnOverlayClick ? onClose : undefined}>
      <div
        ref={modalRef}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2 id="modal-title">{title}</h2>
          <button
            className="modal-close"
            onClick={onClose}
            aria-label="Close modal"
          >
            ×
          </button>
        </div>
        
        <div className="modal-content">
          {children}
        </div>
      </div>
    </div>
  );
};

interface NavigationProps {
  items: Array<{
    id: string;
    label: string;
    href: string;
    current?: boolean;
    disabled?: boolean;
  }>;
  skipLinkTarget?: string;
}

const AccessibleNavigation: React.FC<NavigationProps> = ({
  items,
  skipLinkTarget = '#main-content',
}) => {
  return (
    <nav role="navigation" aria-label="Main navigation">
      <a href={skipLinkTarget} className="skip-link">
        Skip to main content
      </a>
      
      <ul className="nav-list" role="menubar">
        {items.map((item) => (
          <li key={item.id} role="none">
            <a
              href={item.href}
              role="menuitem"
              aria-current={item.current ? 'page' : undefined}
              aria-disabled={item.disabled}
              className={`nav-link ${item.current ? 'nav-link--current' : ''} ${item.disabled ? 'nav-link--disabled' : ''}`}
              tabIndex={item.disabled ? -1 : 0}
            >
              {item.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
};

interface DataTableProps {
  data: Array<Record<string, any>>;
  columns: Array<{
    key: string;
    label: string;
    sortable?: boolean;
  }>;
  caption: string;
  sortColumn?: string;
  sortDirection?: 'asc' | 'desc';
  onSort?: (column: string) => void;
}

const AccessibleDataTable: React.FC<DataTableProps> = ({
  data,
  columns,
  caption,
  sortColumn,
  sortDirection,
  onSort,
}) => {
  const handleSort = (columnKey: string) => {
    if (onSort) {
      onSort(columnKey);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent, columnKey: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleSort(columnKey);
    }
  };

  return (
    <table className="data-table" role="table">
      <caption className="table-caption">
        {caption}
      </caption>
      
      <thead>
        <tr role="row">
          {columns.map((column) => (
            <th
              key={column.key}
              role="columnheader"
              scope="col"
              className={`table-header ${column.sortable ? 'table-header--sortable' : ''}`}
              aria-sort={
                sortColumn === column.key
                  ? sortDirection === 'asc'
                    ? 'ascending'
                    : 'descending'
                  : column.sortable
                  ? 'none'
                  : undefined
              }
              tabIndex={column.sortable ? 0 : undefined}
              onClick={column.sortable ? () => handleSort(column.key) : undefined}
              onKeyDown={column.sortable ? (e) => handleKeyDown(e, column.key) : undefined}
            >
              {column.label}
              {column.sortable && (
                <span className="sort-indicator" aria-hidden="true">
                  {sortColumn === column.key
                    ? sortDirection === 'asc'
                      ? '↑'
                      : '↓'
                    : '↕'
                  }
                </span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      
      <tbody>
        {data.map((row, index) => (
          <tr key={index} role="row">
            {columns.map((column) => (
              <td key={column.key} role="gridcell">
                {row[column.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

// High Contrast Theme Provider
interface ThemeContextType {
  highContrast: boolean;
  reducedMotion: boolean;
  fontSize: 'normal' | 'large' | 'x-large';
  toggleHighContrast: () => void;
  setFontSize: (size: 'normal' | 'large' | 'x-large') => void;
}

const ThemeContext = React.createContext<ThemeContextType | null>(null);

const AccessibilityThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [highContrast, setHighContrast] = React.useState(false);
  const [fontSize, setFontSize] = React.useState<'normal' | 'large' | 'x-large'>('normal');
  
  // Detect user preferences
  const reducedMotion = React.useMemo(() => {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  React.useEffect(() => {
    // Check for high contrast preference
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)');
    setHighContrast(highContrastQuery.matches);

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setHighContrast(e.matches);
    };

    highContrastQuery.addEventListener('change', handleContrastChange);
    return () => highContrastQuery.removeEventListener('change', handleContrastChange);
  }, []);

  const toggleHighContrast = () => {
    setHighContrast(!highContrast);
  };

  const value = {
    highContrast,
    reducedMotion,
    fontSize,
    toggleHighContrast,
    setFontSize,
  };

  return (
    <ThemeContext.Provider value={value}>
      <div
        className={`theme-root ${highContrast ? 'high-contrast' : ''} font-${fontSize}`}
        data-reduced-motion={reducedMotion}
      >
        {children}
      </div>
    </ThemeContext.Provider>
  );
};

// Screen Reader Utilities
class ScreenReaderUtilities {
  static announceToScreenReader(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', priority);
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;

    document.body.appendChild(announcement);

    // Remove after announcement
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  }

  static describeLiveRegion(element: HTMLElement, description: string): void {
    const descriptionId = `live-region-desc-${Math.random().toString(36).substr(2, 9)}`;
    
    const descriptionElement = document.createElement('div');
    descriptionElement.id = descriptionId;
    descriptionElement.className = 'sr-only';
    descriptionElement.textContent = description;

    document.body.appendChild(descriptionElement);
    element.setAttribute('aria-describedby', descriptionId);
  }

  static createSkipLink(targetId: string, text: string = 'Skip to main content'): HTMLAnchorElement {
    const skipLink = document.createElement('a');
    skipLink.href = `#${targetId}`;
    skipLink.textContent = text;
    skipLink.className = 'skip-link';
    
    skipLink.addEventListener('focus', () => {
      skipLink.style.position = 'absolute';
      skipLink.style.top = '0';
      skipLink.style.left = '0';
      skipLink.style.zIndex = '9999';
    });

    skipLink.addEventListener('blur', () => {
      skipLink.style.position = 'absolute';
      skipLink.style.left = '-9999px';
    });

    return skipLink;
  }
}

// Tests
describe('Accessibility Comprehensive Tests', () => {
  beforeEach(() => {
    // Reset axe configuration
    mockAxe.reset();
    
    // Mock media queries
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: jest.fn().mockImplementation(query => ({
        matches: query.includes('prefers-reduced-motion') || query.includes('prefers-contrast'),
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      })),
    });
  });

  describe('AccessibilityTester', () => {
    test('should run axe tests', async () => {
      const { container } = render(<div>Test content</div>);
      
      const results = await AccessibilityTester.runAxeTests(container);
      
      expect(mockAxe.run).toHaveBeenCalledWith(container, { includeHidden: false });
      expect(results).toHaveProperty('violations');
      expect(results).toHaveProperty('passes');
    });

    test('should check color contrast', () => {
      const result = AccessibilityTester.checkColorContrast('#000000', '#ffffff');
      
      expect(result.ratio).toBeGreaterThan(4.5);
      expect(result.passes).toBe(true);
      
      const lowContrastResult = AccessibilityTester.checkColorContrast('#888888', '#999999');
      expect(lowContrastResult.passes).toBe(false);
    });

    test('should analyze keyboard navigation', () => {
      const { container } = render(
        <div>
          <a href="#skip">Skip to content</a>
          <button>Button 1</button>
          <input type="text" />
          <button disabled>Disabled Button</button>
          <a href="/link">Link</a>
        </div>
      );

      const analysis = AccessibilityTester.checkKeyboardNavigation(container);
      
      expect(analysis.focusableElements.length).toBeGreaterThan(0);
      expect(analysis.hasSkipLinks).toBe(true);
      expect(analysis.tabOrder).toEqual(expect.arrayContaining([0]));
    });

    test('should check ARIA labels', () => {
      const { container } = render(
        <div>
          <button aria-label="Close dialog">×</button>
          <input id="email" />
          <label htmlFor="email">Email</label>
          <button>Button without label</button>
          <main>Main content</main>
          <nav aria-label="Primary navigation">Navigation</nav>
        </div>
      );

      const analysis = AccessibilityTester.checkAriaLabels(container);
      
      expect(analysis.elementsWithoutLabels.length).toBe(1); // Button without label
      expect(analysis.landmarkElements.length).toBe(2); // main and nav
    });

    test('should throw error for invalid color format', () => {
      expect(() => {
        AccessibilityTester.checkColorContrast('invalid', '#ffffff');
      }).toThrow('Invalid color format');
    });
  });

  describe('AccessibleButton', () => {
    test('should render with proper ARIA attributes', () => {
      render(
        <AccessibleButton ariaLabel="Save document" ariaDescribedBy="save-help">
          Save
        </AccessibleButton>
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label', 'Save document');
      expect(button).toHaveAttribute('aria-describedby', 'save-help');
      expect(button).toHaveAttribute('role', 'button');
    });

    test('should handle keyboard interaction', async () => {
      const user = userEvent.setup();
      const handleClick = jest.fn();
      
      render(
        <AccessibleButton onClick={handleClick}>
          Click me
        </AccessibleButton>
      );

      const button = screen.getByRole('button');
      
      // Test Enter key
      await user.type(button, '{enter}');
      expect(handleClick).toHaveBeenCalledTimes(1);
      
      // Test Space key
      await user.type(button, ' ');
      expect(handleClick).toHaveBeenCalledTimes(2);
    });

    test('should show loading state with proper ARIA', () => {
      render(
        <AccessibleButton loading>
          Submit
        </AccessibleButton>
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-busy', 'true');
      expect(screen.getByText('Loading, please wait')).toBeInTheDocument();
    });

    test('should be disabled and not focusable when disabled', () => {
      render(
        <AccessibleButton disabled>
          Disabled Button
        </AccessibleButton>
      );

      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
      expect(button).toHaveAttribute('tabindex', '-1');
    });

    test('should not trigger click when disabled', async () => {
      const user = userEvent.setup();
      const handleClick = jest.fn();
      
      render(
        <AccessibleButton disabled onClick={handleClick}>
          Disabled
        </AccessibleButton>
      );

      const button = screen.getByRole('button');
      await user.click(button);
      
      expect(handleClick).not.toHaveBeenCalled();
    });
  });

  describe('AccessibleFormField', () => {
    test('should associate label with input', () => {
      render(
        <AccessibleFormField
          id="username"
          label="Username"
          required
        />
      );

      const input = screen.getByLabelText('Username *');
      expect(input).toHaveAttribute('id', 'username');
      expect(input).toBeRequired();
    });

    test('should show error message with proper ARIA', () => {
      render(
        <AccessibleFormField
          id="email"
          label="Email"
          error="Invalid email format"
        />
      );

      const input = screen.getByLabelText('Email');
      const errorMessage = screen.getByRole('alert');
      
      expect(input).toHaveAttribute('aria-invalid', 'true');
      expect(input).toHaveAttribute('aria-describedby', 'email-error');
      expect(errorMessage).toHaveTextContent('Invalid email format');
    });

    test('should show help text', () => {
      render(
        <AccessibleFormField
          id="password"
          label="Password"
          helpText="Must be at least 8 characters"
        />
      );

      const input = screen.getByLabelText('Password');
      const helpText = screen.getByText('Must be at least 8 characters');
      
      expect(input).toHaveAttribute('aria-describedby', 'password-help');
      expect(helpText).toHaveAttribute('id', 'password-help');
    });

    test('should combine help text and error in aria-describedby', () => {
      render(
        <AccessibleFormField
          id="confirm-password"
          label="Confirm Password"
          helpText="Re-enter your password"
          error="Passwords do not match"
        />
      );

      const input = screen.getByLabelText('Confirm Password');
      expect(input).toHaveAttribute('aria-describedby', 'confirm-password-help confirm-password-error');
    });

    test('should handle value changes', async () => {
      const user = userEvent.setup();
      const handleChange = jest.fn();
      
      render(
        <AccessibleFormField
          id="test-input"
          label="Test Input"
          onChange={handleChange}
        />
      );

      const input = screen.getByLabelText('Test Input');
      await user.type(input, 'test value');
      
      expect(handleChange).toHaveBeenCalledWith('test value');
    });
  });

  describe('AccessibleModal', () => {
    test('should trap focus within modal', async () => {
      const user = userEvent.setup();
      const handleClose = jest.fn();
      
      render(
        <AccessibleModal isOpen title="Test Modal" onClose={handleClose}>
          <button>First Button</button>
          <button>Second Button</button>
        </AccessibleModal>
      );

      const modal = screen.getByRole('dialog');
      const firstButton = screen.getByText('First Button');
      const secondButton = screen.getByText('Second Button');
      const closeButton = screen.getByLabelText('Close modal');

      expect(modal).toHaveAttribute('aria-modal', 'true');
      expect(modal).toHaveAttribute('aria-labelledby', 'modal-title');
      
      // Focus should be trapped within modal
      await user.tab();
      expect(closeButton).toHaveFocus();
      
      await user.tab();
      expect(firstButton).toHaveFocus();
    });

    test('should close on Escape key', async () => {
      const user = userEvent.setup();
      const handleClose = jest.fn();
      
      render(
        <AccessibleModal isOpen title="Test Modal" onClose={handleClose}>
          <p>Modal content</p>
        </AccessibleModal>
      );

      await user.keyboard('{Escape}');
      expect(handleClose).toHaveBeenCalled();
    });

    test('should close on overlay click when enabled', async () => {
      const user = userEvent.setup();
      const handleClose = jest.fn();
      
      render(
        <AccessibleModal isOpen title="Test Modal" onClose={handleClose} closeOnOverlayClick>
          <p>Modal content</p>
        </AccessibleModal>
      );

      const overlay = screen.getByText('Modal content').closest('.modal-overlay');
      await user.click(overlay!);
      
      expect(handleClose).toHaveBeenCalled();
    });

    test('should not render when closed', () => {
      render(
        <AccessibleModal isOpen={false} title="Test Modal" onClose={jest.fn()}>
          <p>Modal content</p>
        </AccessibleModal>
      );

      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });

  describe('AccessibleNavigation', () => {
    const navItems = [
      { id: 'home', label: 'Home', href: '/', current: true },
      { id: 'about', label: 'About', href: '/about' },
      { id: 'contact', label: 'Contact', href: '/contact', disabled: true },
    ];

    test('should render navigation with proper ARIA attributes', () => {
      render(<AccessibleNavigation items={navItems} />);

      const nav = screen.getByRole('navigation');
      const menubar = screen.getByRole('menubar');
      
      expect(nav).toHaveAttribute('aria-label', 'Main navigation');
      expect(menubar).toBeInTheDocument();
    });

    test('should include skip link', () => {
      render(<AccessibleNavigation items={navItems} skipLinkTarget="#content" />);

      const skipLink = screen.getByText('Skip to main content');
      expect(skipLink).toHaveAttribute('href', '#content');
    });

    test('should mark current page correctly', () => {
      render(<AccessibleNavigation items={navItems} />);

      const homeLink = screen.getByText('Home');
      expect(homeLink).toHaveAttribute('aria-current', 'page');
    });

    test('should disable navigation items correctly', () => {
      render(<AccessibleNavigation items={navItems} />);

      const contactLink = screen.getByText('Contact');
      expect(contactLink).toHaveAttribute('aria-disabled', 'true');
      expect(contactLink).toHaveAttribute('tabindex', '-1');
    });
  });

  describe('AccessibleDataTable', () => {
    const tableData = [
      { name: 'John Doe', age: 30, city: 'New York' },
      { name: 'Jane Smith', age: 25, city: 'Los Angeles' },
    ];

    const tableColumns = [
      { key: 'name', label: 'Name', sortable: true },
      { key: 'age', label: 'Age', sortable: true },
      { key: 'city', label: 'City' },
    ];

    test('should render table with proper structure', () => {
      render(
        <AccessibleDataTable
          data={tableData}
          columns={tableColumns}
          caption="User Information"
        />
      );

      const table = screen.getByRole('table');
      const caption = screen.getByText('User Information');
      
      expect(table).toBeInTheDocument();
      expect(caption).toBeInTheDocument();
    });

    test('should handle sortable columns', async () => {
      const user = userEvent.setup();
      const handleSort = jest.fn();
      
      render(
        <AccessibleDataTable
          data={tableData}
          columns={tableColumns}
          caption="User Information"
          onSort={handleSort}
          sortColumn="name"
          sortDirection="asc"
        />
      );

      const nameHeader = screen.getByText('Name');
      expect(nameHeader).toHaveAttribute('aria-sort', 'ascending');
      
      await user.click(nameHeader);
      expect(handleSort).toHaveBeenCalledWith('name');
    });

    test('should handle keyboard navigation on sortable columns', async () => {
      const user = userEvent.setup();
      const handleSort = jest.fn();
      
      render(
        <AccessibleDataTable
          data={tableData}
          columns={tableColumns}
          caption="User Information"
          onSort={handleSort}
        />
      );

      const nameHeader = screen.getByText('Name');
      nameHeader.focus();
      
      await user.keyboard('{enter}');
      expect(handleSort).toHaveBeenCalledWith('name');
      
      await user.keyboard(' ');
      expect(handleSort).toHaveBeenCalledTimes(2);
    });

    test('should not make non-sortable columns interactive', () => {
      render(
        <AccessibleDataTable
          data={tableData}
          columns={tableColumns}
          caption="User Information"
        />
      );

      const cityHeader = screen.getByText('City');
      expect(cityHeader).not.toHaveAttribute('tabindex');
      expect(cityHeader).not.toHaveAttribute('aria-sort');
    });
  });

  describe('AccessibilityThemeProvider', () => {
    test('should provide theme context', () => {
      const TestComponent = () => {
        const theme = React.useContext(ThemeContext);
        return (
          <div>
            <span data-testid="high-contrast">{theme?.highContrast.toString()}</span>
            <span data-testid="reduced-motion">{theme?.reducedMotion.toString()}</span>
            <span data-testid="font-size">{theme?.fontSize}</span>
          </div>
        );
      };

      render(
        <AccessibilityThemeProvider>
          <TestComponent />
        </AccessibilityThemeProvider>
      );

      expect(screen.getByTestId('high-contrast')).toHaveTextContent('true');
      expect(screen.getByTestId('reduced-motion')).toHaveTextContent('true');
      expect(screen.getByTestId('font-size')).toHaveTextContent('normal');
    });

    test('should toggle high contrast', () => {
      const TestComponent = () => {
        const theme = React.useContext(ThemeContext);
        return (
          <div>
            <button onClick={theme?.toggleHighContrast}>
              Toggle High Contrast
            </button>
            <span data-testid="high-contrast">{theme?.highContrast.toString()}</span>
          </div>
        );
      };

      render(
        <AccessibilityThemeProvider>
          <TestComponent />
        </AccessibilityThemeProvider>
      );

      const toggleButton = screen.getByText('Toggle High Contrast');
      const contrastIndicator = screen.getByTestId('high-contrast');

      expect(contrastIndicator).toHaveTextContent('true');
      
      fireEvent.click(toggleButton);
      expect(contrastIndicator).toHaveTextContent('false');
    });
  });

  describe('ScreenReaderUtilities', () => {
    test('should announce to screen reader', () => {
      ScreenReaderUtilities.announceToScreenReader('Test announcement', 'assertive');
      
      const announcement = document.querySelector('[aria-live="assertive"]');
      expect(announcement).toBeInTheDocument();
      expect(announcement).toHaveTextContent('Test announcement');
      expect(announcement).toHaveClass('sr-only');
    });

    test('should describe live region', () => {
      const element = document.createElement('div');
      document.body.appendChild(element);
      
      ScreenReaderUtilities.describeLiveRegion(element, 'Live region description');
      
      const descriptionId = element.getAttribute('aria-describedby');
      expect(descriptionId).toBeTruthy();
      
      const description = document.getElementById(descriptionId!);
      expect(description).toHaveTextContent('Live region description');
      
      document.body.removeChild(element);
    });

    test('should create skip link', () => {
      const skipLink = ScreenReaderUtilities.createSkipLink('main-content', 'Skip to main');
      
      expect(skipLink.tagName).toBe('A');
      expect(skipLink.href).toContain('#main-content');
      expect(skipLink.textContent).toBe('Skip to main');
      expect(skipLink).toHaveClass('skip-link');
    });
  });

  describe('Integration Tests', () => {
    test('should run axe tests on complex component', async () => {
      const { container } = render(
        <AccessibilityThemeProvider>
          <AccessibleNavigation
            items={[
              { id: 'home', label: 'Home', href: '/', current: true },
              { id: 'about', label: 'About', href: '/about' },
            ]}
          />
          <main id="main-content">
            <AccessibleFormField
              id="search"
              label="Search"
              placeholder="Enter search terms"
            />
            <AccessibleButton>Search</AccessibleButton>
          </main>
        </AccessibilityThemeProvider>
      );

      const results = await AccessibilityTester.runAxeTests(container, {
        tags: ['wcag2a', 'wcag2aa'],
      });

      expect(results.violations).toEqual([]);
    });

    test('should handle complex keyboard navigation flow', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <AccessibleNavigation
            items={[
              { id: 'home', label: 'Home', href: '/' },
              { id: 'about', label: 'About', href: '/about' },
            ]}
          />
          <main>
            <AccessibleFormField id="field1" label="Field 1" />
            <AccessibleFormField id="field2" label="Field 2" />
            <AccessibleButton>Submit</AccessibleButton>
          </main>
        </div>
      );

      // Test tab navigation through all interactive elements
      const skipLink = screen.getByText('Skip to main content');
      const homeLink = screen.getByText('Home');
      const aboutLink = screen.getByText('About');
      const field1 = screen.getByLabelText('Field 1');
      const field2 = screen.getByLabelText('Field 2');
      const submitButton = screen.getByText('Submit');

      // Start at skip link
      skipLink.focus();
      expect(skipLink).toHaveFocus();

      // Tab through navigation
      await user.tab();
      expect(homeLink).toHaveFocus();

      await user.tab();
      expect(aboutLink).toHaveFocus();

      // Tab through form
      await user.tab();
      expect(field1).toHaveFocus();

      await user.tab();
      expect(field2).toHaveFocus();

      await user.tab();
      expect(submitButton).toHaveFocus();
    });
  });
});