"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

// =============================================================================
// ACCESSIBILITY CONTEXT & HOOKS
// =============================================================================

interface AccessibilityContextType {
  isHighContrast: boolean;
  reducedMotion: boolean;
  screenReaderMode: boolean;
  focusVisible: boolean;
  announcements: string[];
  announce: (message: string) => void;
  setHighContrast: (enabled: boolean) => void;
}

const AccessibilityContext = React.createContext<AccessibilityContextType | null>(null);

export const useAccessibility = () => {
  const context = React.useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within AccessibilityProvider');
  }
  return context;
};

// =============================================================================
// ACCESSIBILITY PROVIDER
// =============================================================================

export const AccessibilityProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isHighContrast, setIsHighContrast] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [screenReaderMode, setScreenReaderMode] = useState(false);
  const [focusVisible, setFocusVisible] = useState(false);
  const [announcements, setAnnouncements] = useState<string[]>([]);

  // Detect user preferences
  useEffect(() => {
    const mediaQueries = {
      highContrast: window.matchMedia('(prefers-contrast: high)'),
      reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)'),
    };

    setIsHighContrast(mediaQueries.highContrast.matches);
    setReducedMotion(mediaQueries.reducedMotion.matches);

    const handleHighContrastChange = (e: MediaQueryListEvent) => setIsHighContrast(e.matches);
    const handleReducedMotionChange = (e: MediaQueryListEvent) => setReducedMotion(e.matches);

    mediaQueries.highContrast.addEventListener('change', handleHighContrastChange);
    mediaQueries.reducedMotion.addEventListener('change', handleReducedMotionChange);

    return () => {
      mediaQueries.highContrast.removeEventListener('change', handleHighContrastChange);
      mediaQueries.reducedMotion.removeEventListener('change', handleReducedMotionChange);
    };
  }, []);

  // Detect screen reader usage
  useEffect(() => {
    const detectScreenReader = () => {
      const hasScreenReader = 
        navigator.userAgent.includes('NVDA') ||
        navigator.userAgent.includes('JAWS') ||
        navigator.userAgent.includes('VoiceOver') ||
        window.speechSynthesis?.getVoices().length > 0;
      
      setScreenReaderMode(hasScreenReader);
    };

    detectScreenReader();
    setTimeout(detectScreenReader, 1000); // Check again after voices load
  }, []);

  // Focus visible detection
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setFocusVisible(true);
      }
    };

    const handleMouseDown = () => {
      setFocusVisible(false);
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  const announce = (message: string) => {
    setAnnouncements(prev => [...prev, message]);
    setTimeout(() => {
      setAnnouncements(prev => prev.slice(1));
    }, 1000);
  };

  const value: AccessibilityContextType = {
    isHighContrast,
    reducedMotion,
    screenReaderMode,
    focusVisible,
    announcements,
    announce,
    setHighContrast: setIsHighContrast,
  };

  return (
    <AccessibilityContext.Provider value={value}>
      <div 
        className={`accessibility-root ${isHighContrast ? 'high-contrast' : ''} ${reducedMotion ? 'reduced-motion' : ''} ${focusVisible ? 'focus-visible' : ''}`}
      >
        {children}
        <LiveRegion announcements={announcements} />
        <SkipLinks />
      </div>
    </AccessibilityContext.Provider>
  );
};

// =============================================================================
// LIVE REGION FOR ANNOUNCEMENTS
// =============================================================================

const LiveRegion: React.FC<{ announcements: string[] }> = ({ announcements }) => (
  <div
    aria-live="polite"
    aria-atomic="true"
    className="sr-only"
    role="status"
  >
    {announcements.map((announcement, index) => (
      <div key={index}>{announcement}</div>
    ))}
  </div>
);

// =============================================================================
// SKIP LINKS
// =============================================================================

const SkipLinks: React.FC = () => (
  <div className="skip-links">
    <a href="#main-content" className="skip-link">
      Skip to main content
    </a>
    <a href="#navigation" className="skip-link">
      Skip to navigation
    </a>
    <a href="#search" className="skip-link">
      Skip to search
    </a>
  </div>
);

// =============================================================================
// KEYBOARD NAVIGATION HOOK
// =============================================================================

export const useKeyboardNavigation = (
  containerRef: React.RefObject<HTMLElement>,
  options: {
    enableArrowKeys?: boolean;
    enableHomeEnd?: boolean;
    enableTypeAhead?: boolean;
    focusableSelector?: string;
  } = {}
) => {
  const {
    enableArrowKeys = true,
    enableHomeEnd = true,
    enableTypeAhead = false,
    focusableSelector = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  } = options;

  const [currentIndex, setCurrentIndex] = useState(0);
  const typeAheadRef = useRef('');
  const typeAheadTimeoutRef = useRef<number>();

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const focusableElements = Array.from(
        container.querySelectorAll(focusableSelector)
      ) as HTMLElement[];

      if (focusableElements.length === 0) return;

      switch (e.key) {
        case 'ArrowDown':
        case 'ArrowRight':
          if (enableArrowKeys) {
            e.preventDefault();
            const nextIndex = (currentIndex + 1) % focusableElements.length;
            setCurrentIndex(nextIndex);
            focusableElements[nextIndex]?.focus();
          }
          break;

        case 'ArrowUp':
        case 'ArrowLeft':
          if (enableArrowKeys) {
            e.preventDefault();
            const prevIndex = currentIndex === 0 ? focusableElements.length - 1 : currentIndex - 1;
            setCurrentIndex(prevIndex);
            focusableElements[prevIndex]?.focus();
          }
          break;

        case 'Home':
          if (enableHomeEnd) {
            e.preventDefault();
            setCurrentIndex(0);
            focusableElements[0]?.focus();
          }
          break;

        case 'End':
          if (enableHomeEnd) {
            e.preventDefault();
            const lastIndex = focusableElements.length - 1;
            setCurrentIndex(lastIndex);
            focusableElements[lastIndex]?.focus();
          }
          break;

        default:
          if (enableTypeAhead && e.key.length === 1) {
            typeAheadRef.current += e.key.toLowerCase();
            
            const matchingElement = focusableElements.find(el => 
              el.textContent?.toLowerCase().startsWith(typeAheadRef.current)
            );

            if (matchingElement) {
              const matchIndex = focusableElements.indexOf(matchingElement);
              setCurrentIndex(matchIndex);
              matchingElement.focus();
            }

            clearTimeout(typeAheadTimeoutRef.current);
            typeAheadTimeoutRef.current = window.setTimeout(() => {
              typeAheadRef.current = '';
            }, 1000);
          }
          break;
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    return () => {
      container.removeEventListener('keydown', handleKeyDown);
      clearTimeout(typeAheadTimeoutRef.current);
    };
  }, [containerRef, currentIndex, enableArrowKeys, enableHomeEnd, enableTypeAhead, focusableSelector]);

  return { currentIndex, setCurrentIndex };
};

// =============================================================================
// FOCUS TRAP HOOK
// =============================================================================

export const useFocusTrap = (isActive: boolean) => {
  const containerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) as NodeListOf<HTMLElement>;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement?.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement?.focus();
          }
        }
      }

      if (e.key === 'Escape') {
        // Allow parent components to handle escape
        const escapeEvent = new CustomEvent('escapeFocusTrap', { bubbles: true });
        container.dispatchEvent(escapeEvent);
      }
    };

    // Focus first element when trap becomes active
    firstElement?.focus();

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isActive]);

  return containerRef;
};

// =============================================================================
// ACCESSIBLE BUTTON COMPONENT
// =============================================================================

interface AccessibleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  loadingText?: string;
  children: React.ReactNode;
}

export const AccessibleButton: React.FC<AccessibleButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  loadingText = 'Loading...',
  disabled,
  children,
  className = '',
  ...props
}) => {
  const { announce } = useAccessibility();

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    if (loading || disabled) return;
    
    announce(`Button activated: ${typeof children === 'string' ? children : 'Button'}`);
    props.onClick?.(e);
  };

  return (
    <button
      {...props}
      onClick={handleClick}
      disabled={disabled || loading}
      aria-disabled={disabled || loading}
      aria-busy={loading}
      className={`button button-${variant} button-${size} ${className}`}
    >
      {loading ? (
        <>
          <span className="sr-only">{loadingText}</span>
          <div className="w-4 h-4 animate-spin border-2 border-current border-t-transparent rounded-full" />
        </>
      ) : (
        children
      )}
    </button>
  );
};

// =============================================================================
// ACCESSIBLE PANEL COMPONENT
// =============================================================================

interface AccessiblePanelProps {
  title: string;
  id: string;
  children: React.ReactNode;
  expanded?: boolean;
  onToggle?: (expanded: boolean) => void;
  level?: 2 | 3 | 4 | 5 | 6;
}

export const AccessiblePanel: React.FC<AccessiblePanelProps> = ({
  title,
  id,
  children,
  expanded = true,
  onToggle,
  level = 3
}) => {
  const { announce } = useAccessibility();
  const HeadingTag = `h${level}` as keyof JSX.IntrinsicElements;

  const handleToggle = () => {
    const newExpanded = !expanded;
    onToggle?.(newExpanded);
    announce(`Panel ${newExpanded ? 'expanded' : 'collapsed'}: ${title}`);
  };

  return (
    <section className="accessible-panel" aria-labelledby={`${id}-heading`}>
      <HeadingTag id={`${id}-heading`} className="panel-title">
        {onToggle ? (
          <button
            aria-expanded={expanded}
            aria-controls={`${id}-content`}
            onClick={handleToggle}
            className="panel-toggle-button"
          >
            <span className="sr-only">
              {expanded ? 'Collapse' : 'Expand'} panel:
            </span>
            {title}
            <span className={`panel-toggle-icon ${expanded ? 'expanded' : ''}`}>
              ▼
            </span>
          </button>
        ) : (
          title
        )}
      </HeadingTag>
      
      <AnimatePresence>
        {expanded && (
          <motion.div
            id={`${id}-content`}
            className="panel-content"
            role="region"
            aria-labelledby={`${id}-heading`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
};

// =============================================================================
// ACCESSIBILITY STYLES
// =============================================================================

const accessibilityStyles = `
/* Screen reader only content */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Skip links */
.skip-links {
  position: absolute;
  top: -40px;
  left: 6px;
  z-index: 1000;
}

.skip-link {
  position: absolute;
  top: -40px;
  left: 6px;
  background: var(--primary-amber);
  color: var(--bg-primary);
  padding: 8px 16px;
  text-decoration: none;
  border-radius: 4px;
  font-weight: 600;
  transition: top 0.3s;
}

.skip-link:focus {
  top: 6px;
}

/* High contrast mode */
.high-contrast {
  --text-primary: #FFFFFF;
  --text-secondary: #FFFFFF;
  --bg-primary: #000000;
  --bg-secondary: #000000;
  --border-primary: #FFFFFF;
  --primary-amber: #FFFF00;
}

.high-contrast .tiled-panel {
  border-width: 2px;
}

.high-contrast .button {
  border-width: 2px;
}

/* Focus visible */
.focus-visible *:focus {
  outline: 2px solid var(--primary-amber);
  outline-offset: 2px;
}

/* Reduced motion */
.reduced-motion * {
  animation-duration: 0.01ms !important;
  animation-iteration-count: 1 !important;
  transition-duration: 0.01ms !important;
}

/* Panel accessibility */
.accessible-panel {
  margin-bottom: 1rem;
}

.panel-toggle-button {
  background: none;
  border: none;
  color: inherit;
  font: inherit;
  width: 100%;
  text-align: left;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem;
  cursor: pointer;
}

.panel-toggle-button:hover {
  background: rgba(255, 255, 255, 0.1);
}

.panel-toggle-icon {
  transition: transform 0.2s;
}

.panel-toggle-icon.expanded {
  transform: rotate(180deg);
}

/* Button loading state */
.button[aria-busy="true"] {
  cursor: not-allowed;
  opacity: 0.7;
}
`;

// Inject styles
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = accessibilityStyles;
  document.head.appendChild(styleSheet);
}

export default {
  AccessibilityProvider,
  useAccessibility,
  useKeyboardNavigation,
  useFocusTrap,
  AccessibleButton,
  AccessiblePanel
};
