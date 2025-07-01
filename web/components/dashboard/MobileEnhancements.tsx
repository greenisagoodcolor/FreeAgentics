"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, PanInfo } from "framer-motion";
import { Menu, X, Maximize2, Minimize2, MoreVertical } from "lucide-react";

// =============================================================================
// MOBILE DETECTION HOOK
// =============================================================================

export const useMobileDetection = () => {
  const [isMobile, setIsMobile] = useState(false);
  const [isTablet, setIsTablet] = useState(false);
  const [orientation, setOrientation] = useState<"portrait" | "landscape">(
    "portrait",
  );
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const checkDevice = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;

      setScreenSize({ width, height });
      setIsMobile(width < 768);
      setIsTablet(width >= 768 && width < 1024);
      setOrientation(width > height ? "landscape" : "portrait");
    };

    checkDevice();
    window.addEventListener("resize", checkDevice);
    window.addEventListener("orientationchange", checkDevice);

    return () => {
      window.removeEventListener("resize", checkDevice);
      window.removeEventListener("orientationchange", checkDevice);
    };
  }, []);

  return { isMobile, isTablet, orientation, screenSize };
};

// =============================================================================
// MOBILE NAVIGATION DRAWER
// =============================================================================

interface MobileDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
}

export const MobileDrawer: React.FC<MobileDrawerProps> = ({
  isOpen,
  onClose,
  children,
  title = "Navigation",
}) => {
  const drawerRef = useRef<HTMLDivElement>(null);

  // Handle swipe to close
  const handleDragEnd = (event: any, info: PanInfo) => {
    if (info.offset.x < -100 || info.velocity.x < -500) {
      onClose();
    }
  };

  // Handle backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }

    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/50 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleBackdropClick}
          />

          {/* Drawer */}
          <motion.div
            ref={drawerRef}
            className="fixed top-0 left-0 h-full w-80 max-w-[85vw] bg-[var(--bg-secondary)] border-r border-[var(--border-primary)] z-50 overflow-y-auto"
            initial={{ x: -320 }}
            animate={{ x: 0 }}
            exit={{ x: -320 }}
            transition={{ type: "tween", duration: 0.3 }}
            drag="x"
            dragConstraints={{ left: -320, right: 0 }}
            dragElastic={0.2}
            onDragEnd={handleDragEnd}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-[var(--border-primary)]">
              <h2 className="text-lg font-semibold text-[var(--text-primary)]">
                {title}
              </h2>
              <button
                onClick={onClose}
                className="p-2 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="p-4">{children}</div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

// =============================================================================
// MOBILE PANEL STACK
// =============================================================================

interface MobilePanelStackProps {
  panels: Array<{
    id: string;
    title: string;
    content: React.ReactNode;
    icon?: React.ReactNode;
  }>;
  activePanel?: string;
  onPanelChange?: (panelId: string) => void;
}

export const MobilePanelStack: React.FC<MobilePanelStackProps> = ({
  panels,
  activePanel,
  onPanelChange,
}) => {
  const [currentPanel, setCurrentPanel] = useState(
    activePanel || panels[0]?.id,
  );
  const [direction, setDirection] = useState(0);

  const handlePanelChange = (panelId: string) => {
    const currentIndex = panels.findIndex((p) => p.id === currentPanel);
    const newIndex = panels.findIndex((p) => p.id === panelId);

    setDirection(newIndex > currentIndex ? 1 : -1);
    setCurrentPanel(panelId);
    onPanelChange?.(panelId);
  };

  const currentPanelData = panels.find((p) => p.id === currentPanel);

  const variants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 300 : -300,
      opacity: 0,
    }),
    center: {
      zIndex: 1,
      x: 0,
      opacity: 1,
    },
    exit: (direction: number) => ({
      zIndex: 0,
      x: direction < 0 ? 300 : -300,
      opacity: 0,
    }),
  };

  return (
    <div className="mobile-panel-stack h-full flex flex-col">
      {/* Tab Bar */}
      <div className="flex border-b border-[var(--border-primary)] bg-[var(--bg-secondary)]">
        {panels.map((panel) => (
          <button
            key={panel.id}
            onClick={() => handlePanelChange(panel.id)}
            className={`flex-1 flex items-center justify-center gap-2 p-3 text-sm font-medium transition-colors ${
              currentPanel === panel.id
                ? "text-[var(--primary-amber)] border-b-2 border-[var(--primary-amber)]"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            {panel.icon}
            <span className="truncate">{panel.title}</span>
          </button>
        ))}
      </div>

      {/* Panel Content */}
      <div className="flex-1 relative overflow-hidden">
        <AnimatePresence initial={false} custom={direction}>
          <motion.div
            key={currentPanel}
            custom={direction}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{
              x: { type: "spring", stiffness: 300, damping: 30 },
              opacity: { duration: 0.2 },
            }}
            className="absolute inset-0 overflow-auto"
          >
            {currentPanelData?.content}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

// =============================================================================
// MOBILE TILING MANAGER
// =============================================================================

interface MobileTilingManagerProps {
  children: React.ReactNode;
  panels: Array<{
    id: string;
    title: string;
    component: React.ReactNode;
  }>;
}

export const MobileTilingManager: React.FC<MobileTilingManagerProps> = ({
  children,
  panels,
}) => {
  const { isMobile, orientation } = useMobileDetection();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [currentView, setCurrentView] = useState<"stack" | "single">("stack");
  const [selectedPanel, setSelectedPanel] = useState<string | null>(null);

  if (!isMobile) {
    return <>{children}</>;
  }

  const handlePanelSelect = (panelId: string) => {
    setSelectedPanel(panelId);
    setCurrentView("single");
  };

  const handleBackToStack = () => {
    setCurrentView("stack");
    setSelectedPanel(null);
  };

  const selectedPanelData = selectedPanel
    ? panels.find((p) => p.id === selectedPanel)
    : null;

  return (
    <div className="mobile-tiling-manager h-full flex flex-col">
      {/* Mobile Header */}
      <div className="mobile-header h-14 bg-[var(--bg-secondary)] border-b border-[var(--border-primary)] flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsDrawerOpen(true)}
            className="p-2 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-[var(--primary-amber)] flex items-center justify-center">
              <span className="text-xs font-bold text-[var(--bg-primary)]">
                CN
              </span>
            </div>
            <h1 className="text-sm font-semibold text-[var(--text-primary)]">
              {currentView === "single" && selectedPanelData
                ? selectedPanelData.title
                : "Dashboard"}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {currentView === "single" && (
            <button
              onClick={handleBackToStack}
              className="p-2 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
            >
              <Minimize2 className="w-4 h-4" />
            </button>
          )}

          <button className="p-2 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
            <MoreVertical className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {currentView === "stack" ? (
          <MobilePanelStack
            panels={panels.map((panel) => ({
              id: panel.id,
              title: panel.title,
              content: (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-[var(--text-primary)]">
                      {panel.title}
                    </h3>
                    <button
                      onClick={() => handlePanelSelect(panel.id)}
                      className="p-2 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
                      title="Expand"
                    >
                      <Maximize2 className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="max-h-64 overflow-hidden">
                    {panel.component}
                  </div>
                </div>
              ),
            }))}
          />
        ) : (
          selectedPanelData && (
            <div className="h-full p-4">{selectedPanelData.component}</div>
          )
        )}
      </div>

      {/* Mobile Drawer */}
      <MobileDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        title="Navigation"
      >
        <div className="space-y-4">
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-[var(--text-primary)] uppercase tracking-wide">
              Panels
            </h3>
            {panels.map((panel) => (
              <button
                key={panel.id}
                onClick={() => {
                  handlePanelSelect(panel.id);
                  setIsDrawerOpen(false);
                }}
                className="w-full text-left p-3 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              >
                {panel.title}
              </button>
            ))}
          </div>

          <div className="border-t border-[var(--border-primary)] pt-4">
            <h3 className="text-sm font-semibold text-[var(--text-primary)] uppercase tracking-wide mb-2">
              Views
            </h3>
            <div className="space-y-2">
              <button
                onClick={() => {
                  setCurrentView("stack");
                  setIsDrawerOpen(false);
                }}
                className={`w-full text-left p-3 rounded-md transition-colors ${
                  currentView === "stack"
                    ? "bg-[var(--primary-amber)] text-[var(--bg-primary)]"
                    : "hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                }`}
              >
                Stack View
              </button>
              <button
                onClick={() => {
                  if (selectedPanel) {
                    setCurrentView("single");
                  }
                  setIsDrawerOpen(false);
                }}
                disabled={!selectedPanel}
                className={`w-full text-left p-3 rounded-md transition-colors ${
                  currentView === "single"
                    ? "bg-[var(--primary-amber)] text-[var(--bg-primary)]"
                    : selectedPanel
                      ? "hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                      : "opacity-50 cursor-not-allowed text-[var(--text-muted)]"
                }`}
              >
                Single Panel
              </button>
            </div>
          </div>
        </div>
      </MobileDrawer>
    </div>
  );
};

// =============================================================================
// TOUCH GESTURES HOOK
// =============================================================================

interface TouchGestureOptions {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onPinch?: (scale: number) => void;
  threshold?: number;
}

export const useTouchGestures = (
  elementRef: React.RefObject<HTMLElement>,
  options: TouchGestureOptions,
) => {
  const {
    onSwipeLeft,
    onSwipeRight,
    onSwipeUp,
    onSwipeDown,
    onPinch,
    threshold = 50,
  } = options;

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    let startX = 0;
    let startY = 0;
    let startDistance = 0;

    const handleTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
      } else if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        startDistance = Math.sqrt(dx * dx + dy * dy);
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2 && onPinch) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const scale = distance / startDistance;
        onPinch(scale);
      }
    };

    const handleTouchEnd = (e: TouchEvent) => {
      if (e.changedTouches.length === 1) {
        const endX = e.changedTouches[0].clientX;
        const endY = e.changedTouches[0].clientY;

        const deltaX = endX - startX;
        const deltaY = endY - startY;

        const absDeltaX = Math.abs(deltaX);
        const absDeltaY = Math.abs(deltaY);

        if (Math.max(absDeltaX, absDeltaY) > threshold) {
          if (absDeltaX > absDeltaY) {
            // Horizontal swipe
            if (deltaX > 0) {
              onSwipeRight?.();
            } else {
              onSwipeLeft?.();
            }
          } else {
            // Vertical swipe
            if (deltaY > 0) {
              onSwipeDown?.();
            } else {
              onSwipeUp?.();
            }
          }
        }
      }
    };

    element.addEventListener("touchstart", handleTouchStart, { passive: true });
    element.addEventListener("touchmove", handleTouchMove, { passive: true });
    element.addEventListener("touchend", handleTouchEnd, { passive: true });

    return () => {
      element.removeEventListener("touchstart", handleTouchStart);
      element.removeEventListener("touchmove", handleTouchMove);
      element.removeEventListener("touchend", handleTouchEnd);
    };
  }, [
    elementRef,
    onSwipeLeft,
    onSwipeRight,
    onSwipeUp,
    onSwipeDown,
    onPinch,
    threshold,
  ]);
};

export default {
  useMobileDetection,
  MobileDrawer,
  MobilePanelStack,
  MobileTilingManager,
  useTouchGestures,
};
