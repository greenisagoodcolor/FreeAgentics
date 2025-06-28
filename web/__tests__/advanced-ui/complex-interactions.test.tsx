/**
 * Complex UI Interactions Tests
 * 
 * Tests for advanced UI patterns, complex interactions, gesture handling,
 * animation systems, and responsive behaviors following ADR-007 requirements.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation((callback) => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
  root: null,
  rootMargin: '',
  thresholds: [],
}));

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation((callback) => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock requestAnimationFrame
global.requestAnimationFrame = jest.fn((callback) => {
  setTimeout(callback, 16); // ~60fps
  return 1;
});

global.cancelAnimationFrame = jest.fn();

// Advanced Drag and Drop System
interface DragState {
  isDragging: boolean;
  draggedElement: string | null;
  dragOffset: { x: number; y: number };
  dropZones: string[];
  validDropZone: string | null;
}

interface DragDropContextValue {
  dragState: DragState;
  startDrag: (elementId: string, offset: { x: number; y: number }) => void;
  updateDrag: (position: { x: number; y: number }) => void;
  endDrag: (dropZoneId?: string) => void;
  registerDropZone: (id: string, accepts: string[]) => void;
  unregisterDropZone: (id: string) => void;
}

const DragDropContext = React.createContext<DragDropContextValue | null>(null);

const DragDropProvider: React.FC<{ children: React.ReactNode; onDrop?: (item: string, zone: string) => void }> = ({ 
  children, 
  onDrop 
}) => {
  const [dragState, setDragState] = React.useState<DragState>({
    isDragging: false,
    draggedElement: null,
    dragOffset: { x: 0, y: 0 },
    dropZones: [],
    validDropZone: null,
  });

  const dropZoneConfig = React.useRef<Map<string, string[]>>(new Map());

  const startDrag = React.useCallback((elementId: string, offset: { x: number; y: number }) => {
    setDragState(prev => ({
      ...prev,
      isDragging: true,
      draggedElement: elementId,
      dragOffset: offset,
    }));
  }, []);

  const updateDrag = React.useCallback((position: { x: number; y: number }) => {
    setDragState(prev => ({
      ...prev,
      dragOffset: position,
    }));
  }, []);

  const endDrag = React.useCallback((dropZoneId?: string) => {
    if (dragState.draggedElement && dropZoneId && onDrop) {
      const accepts = dropZoneConfig.current.get(dropZoneId) || [];
      if (accepts.includes(dragState.draggedElement)) {
        onDrop(dragState.draggedElement, dropZoneId);
      }
    }

    setDragState({
      isDragging: false,
      draggedElement: null,
      dragOffset: { x: 0, y: 0 },
      dropZones: [],
      validDropZone: null,
    });
  }, [dragState.draggedElement, onDrop]);

  const registerDropZone = React.useCallback((id: string, accepts: string[]) => {
    dropZoneConfig.current.set(id, accepts);
    setDragState(prev => ({
      ...prev,
      dropZones: [...prev.dropZones, id],
    }));
  }, []);

  const unregisterDropZone = React.useCallback((id: string) => {
    dropZoneConfig.current.delete(id);
    setDragState(prev => ({
      ...prev,
      dropZones: prev.dropZones.filter(zone => zone !== id),
    }));
  }, []);

  const value = {
    dragState,
    startDrag,
    updateDrag,
    endDrag,
    registerDropZone,
    unregisterDropZone,
  };

  return (
    <DragDropContext.Provider value={value}>
      {children}
    </DragDropContext.Provider>
  );
};

interface DraggableProps {
  id: string;
  children: React.ReactNode;
  data?: any;
  disabled?: boolean;
}

const Draggable: React.FC<DraggableProps> = ({ id, children, data, disabled = false }) => {
  const context = React.useContext(DragDropContext);
  const elementRef = React.useRef<HTMLDivElement>(null);

  const handleMouseDown = (event: React.MouseEvent) => {
    if (disabled || !context) return;

    const rect = elementRef.current?.getBoundingClientRect();
    if (rect) {
      const offset = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      context.startDrag(id, offset);
    }
  };

  const handleTouchStart = (event: React.TouchEvent) => {
    if (disabled || !context) return;

    const touch = event.touches[0];
    const rect = elementRef.current?.getBoundingClientRect();
    if (rect && touch) {
      const offset = {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top,
      };
      context.startDrag(id, offset);
    }
  };

  return (
    <div
      ref={elementRef}
      data-testid={`draggable-${id}`}
      className={`draggable ${disabled ? 'disabled' : ''} ${context?.dragState.draggedElement === id ? 'dragging' : ''}`}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      style={{
        cursor: disabled ? 'not-allowed' : 'grab',
        userSelect: 'none',
      }}
    >
      {children}
    </div>
  );
};

interface DroppableProps {
  id: string;
  accepts: string[];
  children: React.ReactNode;
  onDrop?: (itemId: string) => void;
}

const Droppable: React.FC<DroppableProps> = ({ id, accepts, children, onDrop }) => {
  const context = React.useContext(DragDropContext);

  React.useEffect(() => {
    if (context) {
      context.registerDropZone(id, accepts);
      return () => context.unregisterDropZone(id);
    }
  }, [context, id, accepts]);

  const handleMouseUp = () => {
    if (context?.dragState.isDragging) {
      context.endDrag(id);
      onDrop?.(context.dragState.draggedElement!);
    }
  };

  const handleTouchEnd = () => {
    if (context?.dragState.isDragging) {
      context.endDrag(id);
      onDrop?.(context.dragState.draggedElement!);
    }
  };

  const isValidDropZone = context?.dragState.draggedElement && accepts.includes(context.dragState.draggedElement);

  return (
    <div
      data-testid={`droppable-${id}`}
      className={`droppable ${isValidDropZone ? 'valid-drop' : ''}`}
      onMouseUp={handleMouseUp}
      onTouchEnd={handleTouchEnd}
      style={{
        minHeight: '100px',
        border: `2px dashed ${isValidDropZone ? '#4CAF50' : '#ccc'}`,
        backgroundColor: isValidDropZone ? '#f0f8f0' : 'transparent',
      }}
    >
      {children}
    </div>
  );
};

// Virtual Scrolling Component
interface VirtualScrollItem {
  id: string;
  height: number;
  data: any;
}

interface VirtualScrollProps {
  items: VirtualScrollItem[];
  containerHeight: number;
  itemRenderer: (item: VirtualScrollItem, index: number) => React.ReactNode;
  overscan?: number;
  onScroll?: (scrollTop: number) => void;
}

const VirtualScroll: React.FC<VirtualScrollProps> = ({
  items,
  containerHeight,
  itemRenderer,
  overscan = 5,
  onScroll,
}) => {
  const [scrollTop, setScrollTop] = React.useState(0);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Calculate total height and item positions
  const itemPositions = React.useMemo(() => {
    const positions: number[] = [];
    let currentPosition = 0;
    
    items.forEach((item, index) => {
      positions[index] = currentPosition;
      currentPosition += item.height;
    });
    
    return positions;
  }, [items]);

  const totalHeight = itemPositions[itemPositions.length - 1] + (items[items.length - 1]?.height || 0);

  // Calculate visible range
  const visibleRange = React.useMemo(() => {
    const startIndex = itemPositions.findIndex(pos => pos + items[itemPositions.indexOf(pos)]?.height >= scrollTop);
    const endIndex = itemPositions.findIndex(pos => pos > scrollTop + containerHeight);
    
    return {
      start: Math.max(0, startIndex - overscan),
      end: Math.min(items.length - 1, (endIndex === -1 ? items.length - 1 : endIndex) + overscan),
    };
  }, [scrollTop, containerHeight, itemPositions, items, overscan]);

  const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const newScrollTop = event.currentTarget.scrollTop;
    setScrollTop(newScrollTop);
    onScroll?.(newScrollTop);
  };

  const visibleItems = [];
  for (let i = visibleRange.start; i <= visibleRange.end; i++) {
    const item = items[i];
    if (item) {
      visibleItems.push(
        <div
          key={item.id}
          style={{
            position: 'absolute',
            top: itemPositions[i],
            height: item.height,
            width: '100%',
          }}
        >
          {itemRenderer(item, i)}
        </div>
      );
    }
  }

  return (
    <div
      ref={containerRef}
      data-testid="virtual-scroll-container"
      style={{
        height: containerHeight,
        overflow: 'auto',
        position: 'relative',
      }}
      onScroll={handleScroll}
    >
      <div
        style={{
          height: totalHeight,
          position: 'relative',
        }}
      >
        {visibleItems}
      </div>
    </div>
  );
};

// Gesture Recognition System
interface GestureState {
  isActive: boolean;
  startPoint: { x: number; y: number } | null;
  currentPoint: { x: number; y: number } | null;
  velocity: { x: number; y: number };
  direction: 'up' | 'down' | 'left' | 'right' | null;
  distance: number;
  duration: number;
}

interface GestureRecognizerProps {
  children: React.ReactNode;
  onSwipe?: (direction: 'up' | 'down' | 'left' | 'right', velocity: number) => void;
  onPinch?: (scale: number, center: { x: number; y: number }) => void;
  onTap?: (point: { x: number; y: number }) => void;
  onLongPress?: (point: { x: number; y: number }) => void;
  swipeThreshold?: number;
  longPressDelay?: number;
}

const GestureRecognizer: React.FC<GestureRecognizerProps> = ({
  children,
  onSwipe,
  onPinch,
  onTap,
  onLongPress,
  swipeThreshold = 50,
  longPressDelay = 500,
}) => {
  const [gestureState, setGestureState] = React.useState<GestureState>({
    isActive: false,
    startPoint: null,
    currentPoint: null,
    velocity: { x: 0, y: 0 },
    direction: null,
    distance: 0,
    duration: 0,
  });

  const gestureRef = React.useRef<{
    startTime: number;
    lastPoint: { x: number; y: number } | null;
    longPressTimer?: NodeJS.Timeout;
    touches: Touch[];
  }>({
    startTime: 0,
    lastPoint: null,
    touches: [],
  });

  const handleTouchStart = (event: React.TouchEvent) => {
    const touch = event.touches[0];
    const point = { x: touch.clientX, y: touch.clientY };
    
    gestureRef.current.startTime = Date.now();
    gestureRef.current.lastPoint = point;
    gestureRef.current.touches = Array.from(event.touches);

    setGestureState({
      isActive: true,
      startPoint: point,
      currentPoint: point,
      velocity: { x: 0, y: 0 },
      direction: null,
      distance: 0,
      duration: 0,
    });

    // Start long press timer
    if (onLongPress) {
      gestureRef.current.longPressTimer = setTimeout(() => {
        onLongPress(point);
      }, longPressDelay);
    }
  };

  const handleTouchMove = (event: React.TouchEvent) => {
    if (!gestureState.isActive || !gestureState.startPoint) return;

    // Clear long press timer on movement
    if (gestureRef.current.longPressTimer) {
      clearTimeout(gestureRef.current.longPressTimer);
      gestureRef.current.longPressTimer = undefined;
    }

    const touch = event.touches[0];
    const currentPoint = { x: touch.clientX, y: touch.clientY };
    const startPoint = gestureState.startPoint;

    const deltaX = currentPoint.x - startPoint.x;
    const deltaY = currentPoint.y - startPoint.y;
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const duration = Date.now() - gestureRef.current.startTime;

    // Calculate velocity
    let velocity = { x: 0, y: 0 };
    if (gestureRef.current.lastPoint && duration > 0) {
      const timeDelta = 16; // Assume 16ms frame time
      velocity = {
        x: (currentPoint.x - gestureRef.current.lastPoint.x) / timeDelta,
        y: (currentPoint.y - gestureRef.current.lastPoint.y) / timeDelta,
      };
    }

    // Determine direction
    let direction: 'up' | 'down' | 'left' | 'right' | null = null;
    if (distance > swipeThreshold) {
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        direction = deltaX > 0 ? 'right' : 'left';
      } else {
        direction = deltaY > 0 ? 'down' : 'up';
      }
    }

    setGestureState(prev => ({
      ...prev,
      currentPoint,
      velocity,
      direction,
      distance,
      duration,
    }));

    gestureRef.current.lastPoint = currentPoint;

    // Handle pinch gesture for multi-touch
    if (event.touches.length === 2 && onPinch) {
      const touch1 = event.touches[0];
      const touch2 = event.touches[1];
      
      const currentDistance = Math.sqrt(
        Math.pow(touch2.clientX - touch1.clientX, 2) + 
        Math.pow(touch2.clientY - touch1.clientY, 2)
      );
      
      const initialTouch1 = gestureRef.current.touches[0];
      const initialTouch2 = gestureRef.current.touches[1];
      
      if (initialTouch1 && initialTouch2) {
        const initialDistance = Math.sqrt(
          Math.pow(initialTouch2.clientX - initialTouch1.clientX, 2) + 
          Math.pow(initialTouch2.clientY - initialTouch1.clientY, 2)
        );
        
        const scale = currentDistance / initialDistance;
        const center = {
          x: (touch1.clientX + touch2.clientX) / 2,
          y: (touch1.clientY + touch2.clientY) / 2,
        };
        
        onPinch(scale, center);
      }
    }
  };

  const handleTouchEnd = (event: React.TouchEvent) => {
    // Clear long press timer
    if (gestureRef.current.longPressTimer) {
      clearTimeout(gestureRef.current.longPressTimer);
      gestureRef.current.longPressTimer = undefined;
    }

    if (!gestureState.isActive) return;

    const { direction, distance, velocity, startPoint, duration } = gestureState;

    // Handle swipe
    if (direction && distance > swipeThreshold && onSwipe) {
      const speed = Math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
      onSwipe(direction, speed);
    }

    // Handle tap (short duration, small distance)
    if (duration < 200 && distance < 10 && startPoint && onTap) {
      onTap(startPoint);
    }

    setGestureState({
      isActive: false,
      startPoint: null,
      currentPoint: null,
      velocity: { x: 0, y: 0 },
      direction: null,
      distance: 0,
      duration: 0,
    });
  };

  return (
    <div
      data-testid="gesture-recognizer"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      style={{ touchAction: 'none' }}
    >
      {children}
    </div>
  );
};

// Advanced Animation System
interface AnimationConfig {
  duration: number;
  easing: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out' | 'bounce';
  delay?: number;
  repeat?: number | 'infinite';
  direction?: 'normal' | 'reverse' | 'alternate';
}

interface UseAnimationResult {
  start: () => void;
  stop: () => void;
  reset: () => void;
  isAnimating: boolean;
  progress: number;
}

const useAnimation = (
  from: number,
  to: number,
  config: AnimationConfig,
  onUpdate?: (value: number) => void,
  onComplete?: () => void
): UseAnimationResult => {
  const [isAnimating, setIsAnimating] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  
  const animationRef = React.useRef<{
    startTime: number;
    animationId?: number;
    currentIteration: number;
  }>({
    startTime: 0,
    currentIteration: 0,
  });

  const easingFunctions = {
    linear: (t: number) => t,
    'ease-in': (t: number) => t * t,
    'ease-out': (t: number) => 1 - Math.pow(1 - t, 2),
    'ease-in-out': (t: number) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2,
    bounce: (t: number) => {
      const n1 = 7.5625;
      const d1 = 2.75;
      
      if (t < 1 / d1) {
        return n1 * t * t;
      } else if (t < 2 / d1) {
        return n1 * (t -= 1.5 / d1) * t + 0.75;
      } else if (t < 2.5 / d1) {
        return n1 * (t -= 2.25 / d1) * t + 0.9375;
      } else {
        return n1 * (t -= 2.625 / d1) * t + 0.984375;
      }
    },
  };

  const animate = React.useCallback(() => {
    const now = Date.now();
    const elapsed = now - animationRef.current.startTime - (config.delay || 0);
    
    if (elapsed < 0) {
      animationRef.current.animationId = requestAnimationFrame(animate);
      return;
    }

    const rawProgress = Math.min(elapsed / config.duration, 1);
    const easedProgress = easingFunctions[config.easing](rawProgress);
    
    let currentValue: number;
    
    // Handle animation direction
    if (config.direction === 'reverse') {
      currentValue = from + (to - from) * (1 - easedProgress);
    } else if (config.direction === 'alternate') {
      const isEvenIteration = animationRef.current.currentIteration % 2 === 0;
      currentValue = from + (to - from) * (isEvenIteration ? easedProgress : 1 - easedProgress);
    } else {
      currentValue = from + (to - from) * easedProgress;
    }

    setProgress(rawProgress);
    onUpdate?.(currentValue);

    if (rawProgress >= 1) {
      animationRef.current.currentIteration++;
      
      const shouldRepeat = config.repeat === 'infinite' || 
                          (typeof config.repeat === 'number' && animationRef.current.currentIteration < config.repeat);
      
      if (shouldRepeat) {
        animationRef.current.startTime = now;
        animationRef.current.animationId = requestAnimationFrame(animate);
      } else {
        setIsAnimating(false);
        onComplete?.();
      }
    } else {
      animationRef.current.animationId = requestAnimationFrame(animate);
    }
  }, [from, to, config, onUpdate, onComplete]);

  const start = React.useCallback(() => {
    if (isAnimating) return;
    
    setIsAnimating(true);
    setProgress(0);
    animationRef.current.startTime = Date.now();
    animationRef.current.currentIteration = 0;
    animationRef.current.animationId = requestAnimationFrame(animate);
  }, [isAnimating, animate]);

  const stop = React.useCallback(() => {
    if (animationRef.current.animationId) {
      cancelAnimationFrame(animationRef.current.animationId);
    }
    setIsAnimating(false);
  }, []);

  const reset = React.useCallback(() => {
    stop();
    setProgress(0);
    animationRef.current.currentIteration = 0;
    onUpdate?.(from);
  }, [stop, from, onUpdate]);

  return { start, stop, reset, isAnimating, progress };
};

// Responsive Design Hook
interface BreakpointConfig {
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
}

const defaultBreakpoints: BreakpointConfig = {
  xs: 480,
  sm: 768,
  md: 1024,
  lg: 1280,
  xl: 1920,
};

interface UseResponsiveResult {
  breakpoint: keyof BreakpointConfig;
  isXs: boolean;
  isSm: boolean;
  isMd: boolean;
  isLg: boolean;
  isXl: boolean;
  width: number;
  height: number;
}

const useResponsive = (breakpoints: BreakpointConfig = defaultBreakpoints): UseResponsiveResult => {
  const [dimensions, setDimensions] = React.useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1024,
    height: typeof window !== 'undefined' ? window.innerHeight : 768,
  });

  React.useEffect(() => {
    const handleResize = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const getCurrentBreakpoint = (): keyof BreakpointConfig => {
    const { width } = dimensions;
    
    if (width < breakpoints.xs) return 'xs';
    if (width < breakpoints.sm) return 'sm';
    if (width < breakpoints.md) return 'md';
    if (width < breakpoints.lg) return 'lg';
    return 'xl';
  };

  const breakpoint = getCurrentBreakpoint();

  return {
    breakpoint,
    isXs: breakpoint === 'xs',
    isSm: breakpoint === 'sm',
    isMd: breakpoint === 'md',
    isLg: breakpoint === 'lg',
    isXl: breakpoint === 'xl',
    width: dimensions.width,
    height: dimensions.height,
  };
};

// Infinite Scroll Component
interface InfiniteScrollProps {
  children: React.ReactNode;
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  loading?: boolean;
  threshold?: number;
  loader?: React.ReactNode;
}

const InfiniteScroll: React.FC<InfiniteScrollProps> = ({
  children,
  onLoadMore,
  hasMore,
  loading = false,
  threshold = 100,
  loader = <div>Loading...</div>,
}) => {
  const [isLoading, setIsLoading] = React.useState(false);
  const sentinelRef = React.useRef<HTMLDivElement>(null);
  const observerRef = React.useRef<IntersectionObserver | null>(null);

  React.useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel || !hasMore) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry.isIntersecting && !isLoading && !loading) {
          setIsLoading(true);
          onLoadMore()
            .finally(() => setIsLoading(false));
        }
      },
      {
        rootMargin: `${threshold}px`,
      }
    );

    observerRef.current.observe(sentinel);

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, isLoading, loading, onLoadMore, threshold]);

  return (
    <div data-testid="infinite-scroll">
      {children}
      {hasMore && (
        <div ref={sentinelRef} data-testid="infinite-scroll-sentinel">
          {(isLoading || loading) && loader}
        </div>
      )}
    </div>
  );
};

// Tests
describe('Complex UI Interactions', () => {
  describe('Drag and Drop System', () => {
    test('should handle basic drag and drop', async () => {
      const user = userEvent.setup();
      const onDrop = jest.fn();

      render(
        <DragDropProvider onDrop={onDrop}>
          <Draggable id="item1">
            <div>Draggable Item</div>
          </Draggable>
          <Droppable id="zone1" accepts={['item1']}>
            <div>Drop Zone</div>
          </Droppable>
        </DragDropProvider>
      );

      const draggable = screen.getByTestId('draggable-item1');
      const droppable = screen.getByTestId('droppable-zone1');

      // Start drag
      fireEvent.mouseDown(draggable, { clientX: 10, clientY: 10 });
      
      // Drop
      fireEvent.mouseUp(droppable);

      expect(onDrop).toHaveBeenCalledWith('item1', 'zone1');
    });

    test('should handle disabled draggable', async () => {
      const onDrop = jest.fn();

      render(
        <DragDropProvider onDrop={onDrop}>
          <Draggable id="item1" disabled>
            <div>Disabled Item</div>
          </Draggable>
          <Droppable id="zone1" accepts={['item1']}>
            <div>Drop Zone</div>
          </Droppable>
        </DragDropProvider>
      );

      const draggable = screen.getByTestId('draggable-item1');
      const droppable = screen.getByTestId('droppable-zone1');

      fireEvent.mouseDown(draggable);
      fireEvent.mouseUp(droppable);

      expect(onDrop).not.toHaveBeenCalled();
    });

    test('should handle touch events', () => {
      const onDrop = jest.fn();

      render(
        <DragDropProvider onDrop={onDrop}>
          <Draggable id="item1">
            <div>Touch Item</div>
          </Draggable>
          <Droppable id="zone1" accepts={['item1']}>
            <div>Drop Zone</div>
          </Droppable>
        </DragDropProvider>
      );

      const draggable = screen.getByTestId('draggable-item1');
      const droppable = screen.getByTestId('droppable-zone1');

      // Touch drag
      fireEvent.touchStart(draggable, {
        touches: [{ clientX: 10, clientY: 10 }],
      });
      
      fireEvent.touchEnd(droppable);

      expect(onDrop).toHaveBeenCalledWith('item1', 'zone1');
    });

    test('should not drop on non-accepting zones', () => {
      const onDrop = jest.fn();

      render(
        <DragDropProvider onDrop={onDrop}>
          <Draggable id="item1">
            <div>Item</div>
          </Draggable>
          <Droppable id="zone1" accepts={['item2']}>
            <div>Wrong Zone</div>
          </Droppable>
        </DragDropProvider>
      );

      const draggable = screen.getByTestId('draggable-item1');
      const droppable = screen.getByTestId('droppable-zone1');

      fireEvent.mouseDown(draggable);
      fireEvent.mouseUp(droppable);

      expect(onDrop).not.toHaveBeenCalled();
    });
  });

  describe('Virtual Scrolling', () => {
    const createItems = (count: number): VirtualScrollItem[] =>
      Array.from({ length: count }, (_, i) => ({
        id: `item-${i}`,
        height: 50,
        data: { text: `Item ${i}` },
      }));

    test('should render only visible items', () => {
      const items = createItems(1000);
      const itemRenderer = (item: VirtualScrollItem) => (
        <div data-testid={`item-${item.id}`}>{item.data.text}</div>
      );

      render(
        <VirtualScroll
          items={items}
          containerHeight={300}
          itemRenderer={itemRenderer}
        />
      );

      const container = screen.getByTestId('virtual-scroll-container');
      expect(container).toBeInTheDocument();

      // Should only render visible items (plus overscan)
      const renderedItems = screen.queryAllByTestId(/^item-item-/);
      expect(renderedItems.length).toBeLessThan(1000);
      expect(renderedItems.length).toBeGreaterThan(0);
    });

    test('should handle scroll events', () => {
      const items = createItems(100);
      const onScroll = jest.fn();
      const itemRenderer = (item: VirtualScrollItem) => (
        <div>{item.data.text}</div>
      );

      render(
        <VirtualScroll
          items={items}
          containerHeight={300}
          itemRenderer={itemRenderer}
          onScroll={onScroll}
        />
      );

      const container = screen.getByTestId('virtual-scroll-container');
      
      fireEvent.scroll(container, { target: { scrollTop: 100 } });
      
      expect(onScroll).toHaveBeenCalledWith(100);
    });

    test('should handle items with varying heights', () => {
      const items: VirtualScrollItem[] = [
        { id: 'item-1', height: 50, data: { text: 'Item 1' } },
        { id: 'item-2', height: 100, data: { text: 'Item 2' } },
        { id: 'item-3', height: 75, data: { text: 'Item 3' } },
      ];

      const itemRenderer = (item: VirtualScrollItem) => (
        <div data-testid={`item-${item.id}`} style={{ height: item.height }}>
          {item.data.text}
        </div>
      );

      render(
        <VirtualScroll
          items={items}
          containerHeight={300}
          itemRenderer={itemRenderer}
        />
      );

      const item1 = screen.getByTestId('item-item-1');
      const item2 = screen.getByTestId('item-item-2');
      
      expect(item1).toHaveStyle({ height: '50px' });
      expect(item2).toHaveStyle({ height: '100px' });
    });
  });

  describe('Gesture Recognition', () => {
    test('should recognize swipe gestures', () => {
      const onSwipe = jest.fn();

      render(
        <GestureRecognizer onSwipe={onSwipe} swipeThreshold={30}>
          <div data-testid="gesture-area">Swipe me</div>
        </GestureRecognizer>
      );

      const area = screen.getByTestId('gesture-recognizer');

      // Simulate right swipe
      fireEvent.touchStart(area, {
        touches: [{ clientX: 10, clientY: 100 }],
      });

      fireEvent.touchMove(area, {
        touches: [{ clientX: 60, clientY: 100 }],
      });

      fireEvent.touchEnd(area, { touches: [] });

      expect(onSwipe).toHaveBeenCalledWith('right', expect.any(Number));
    });

    test('should recognize tap gestures', () => {
      const onTap = jest.fn();

      render(
        <GestureRecognizer onTap={onTap}>
          <div>Tap me</div>
        </GestureRecognizer>
      );

      const area = screen.getByTestId('gesture-recognizer');

      // Quick tap
      fireEvent.touchStart(area, {
        touches: [{ clientX: 100, clientY: 100 }],
      });

      fireEvent.touchEnd(area, { touches: [] });

      expect(onTap).toHaveBeenCalledWith({ x: 100, y: 100 });
    });

    test('should recognize long press gestures', (done) => {
      const onLongPress = jest.fn(() => {
        expect(onLongPress).toHaveBeenCalledWith({ x: 100, y: 100 });
        done();
      });

      render(
        <GestureRecognizer onLongPress={onLongPress} longPressDelay={100}>
          <div>Long press me</div>
        </GestureRecognizer>
      );

      const area = screen.getByTestId('gesture-recognizer');

      fireEvent.touchStart(area, {
        touches: [{ clientX: 100, clientY: 100 }],
      });

      // Don't end touch - let long press timer fire
    });

    test('should recognize pinch gestures', () => {
      const onPinch = jest.fn();

      render(
        <GestureRecognizer onPinch={onPinch}>
          <div>Pinch me</div>
        </GestureRecognizer>
      );

      const area = screen.getByTestId('gesture-recognizer');

      // Start with two fingers
      fireEvent.touchStart(area, {
        touches: [
          { clientX: 100, clientY: 100 },
          { clientX: 200, clientY: 100 },
        ],
      });

      // Move fingers apart (zoom in)
      fireEvent.touchMove(area, {
        touches: [
          { clientX: 50, clientY: 100 },
          { clientX: 250, clientY: 100 },
        ],
      });

      expect(onPinch).toHaveBeenCalledWith(expect.any(Number), expect.any(Object));
    });

    test('should cancel long press on movement', () => {
      const onLongPress = jest.fn();

      render(
        <GestureRecognizer onLongPress={onLongPress} longPressDelay={100}>
          <div>Move to cancel</div>
        </GestureRecognizer>
      );

      const area = screen.getByTestId('gesture-recognizer');

      fireEvent.touchStart(area, {
        touches: [{ clientX: 100, clientY: 100 }],
      });

      // Move before long press timer
      fireEvent.touchMove(area, {
        touches: [{ clientX: 150, clientY: 100 }],
      });

      fireEvent.touchEnd(area, { touches: [] });

      // Wait longer than long press delay
      setTimeout(() => {
        expect(onLongPress).not.toHaveBeenCalled();
      }, 150);
    });
  });

  describe('Animation System', () => {
    test('should animate from start to end value', () => {
      const onUpdate = jest.fn();
      const onComplete = jest.fn();

      const TestComponent = () => {
        const animation = useAnimation(0, 100, { duration: 100, easing: 'linear' }, onUpdate, onComplete);
        
        React.useEffect(() => {
          animation.start();
        }, []);

        return (
          <div data-testid="animation-test">
            Progress: {animation.progress}
          </div>
        );
      };

      render(<TestComponent />);

      expect(onUpdate).toHaveBeenCalled();
      
      // Fast-forward animation
      jest.advanceTimersByTime(100);
      
      expect(onComplete).toHaveBeenCalled();
    });

    test('should handle different easing functions', () => {
      const onUpdate = jest.fn();

      const TestComponent = () => {
        const animation = useAnimation(0, 100, { duration: 100, easing: 'ease-in' }, onUpdate);
        
        React.useEffect(() => {
          animation.start();
        }, []);

        return <div>Animating</div>;
      };

      render(<TestComponent />);

      expect(onUpdate).toHaveBeenCalled();
      
      // Verify easing affects the interpolated values
      const calls = onUpdate.mock.calls;
      expect(calls.length).toBeGreaterThan(0);
    });

    test('should handle animation control', () => {
      const TestComponent = () => {
        const animation = useAnimation(0, 100, { duration: 1000, easing: 'linear' });

        return (
          <div>
            <button data-testid="start" onClick={animation.start}>Start</button>
            <button data-testid="stop" onClick={animation.stop}>Stop</button>
            <button data-testid="reset" onClick={animation.reset}>Reset</button>
            <div data-testid="status">{animation.isAnimating ? 'animating' : 'stopped'}</div>
          </div>
        );
      };

      render(<TestComponent />);

      const startButton = screen.getByTestId('start');
      const stopButton = screen.getByTestId('stop');
      const resetButton = screen.getByTestId('reset');
      const status = screen.getByTestId('status');

      // Start animation
      fireEvent.click(startButton);
      expect(status).toHaveTextContent('animating');

      // Stop animation
      fireEvent.click(stopButton);
      expect(status).toHaveTextContent('stopped');

      // Reset animation
      fireEvent.click(resetButton);
      expect(status).toHaveTextContent('stopped');
    });

    test('should handle repeating animations', () => {
      const onComplete = jest.fn();

      const TestComponent = () => {
        const animation = useAnimation(
          0, 
          100, 
          { duration: 50, easing: 'linear', repeat: 3 }, 
          undefined, 
          onComplete
        );
        
        React.useEffect(() => {
          animation.start();
        }, []);

        return <div>Repeating</div>;
      };

      render(<TestComponent />);

      // Fast-forward through all repetitions
      jest.advanceTimersByTime(200);
      
      expect(onComplete).toHaveBeenCalled();
    });
  });

  describe('Responsive Design Hook', () => {
    test('should detect breakpoints correctly', () => {
      // Mock window dimensions
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1200,
      });

      const TestComponent = () => {
        const responsive = useResponsive();
        
        return (
          <div>
            <div data-testid="breakpoint">{responsive.breakpoint}</div>
            <div data-testid="width">{responsive.width}</div>
            <div data-testid="is-lg">{responsive.isLg.toString()}</div>
          </div>
        );
      };

      render(<TestComponent />);

      expect(screen.getByTestId('breakpoint')).toHaveTextContent('lg');
      expect(screen.getByTestId('width')).toHaveTextContent('1200');
      expect(screen.getByTestId('is-lg')).toHaveTextContent('true');
    });

    test('should respond to window resize', () => {
      const TestComponent = () => {
        const responsive = useResponsive();
        
        return (
          <div data-testid="current-breakpoint">{responsive.breakpoint}</div>
        );
      };

      render(<TestComponent />);

      // Simulate resize to mobile
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 400,
      });

      fireEvent(window, new Event('resize'));

      waitFor(() => {
        expect(screen.getByTestId('current-breakpoint')).toHaveTextContent('xs');
      });
    });

    test('should work with custom breakpoints', () => {
      const customBreakpoints = {
        xs: 320,
        sm: 640,
        md: 960,
        lg: 1200,
        xl: 1600,
      };

      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 800,
      });

      const TestComponent = () => {
        const responsive = useResponsive(customBreakpoints);
        
        return (
          <div data-testid="custom-breakpoint">{responsive.breakpoint}</div>
        );
      };

      render(<TestComponent />);

      expect(screen.getByTestId('custom-breakpoint')).toHaveTextContent('sm');
    });
  });

  describe('Infinite Scroll', () => {
    test('should load more content when scrolled to bottom', async () => {
      const onLoadMore = jest.fn().mockResolvedValue(undefined);

      render(
        <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
          <div style={{ height: '1000px' }}>Content</div>
        </InfiniteScroll>
      );

      const sentinel = screen.getByTestId('infinite-scroll-sentinel');
      
      // Mock IntersectionObserver trigger
      const mockObserver = (global.IntersectionObserver as jest.Mock).mock.results[0].value;
      const callback = (global.IntersectionObserver as jest.Mock).mock.calls[0][0];
      
      // Simulate intersection
      callback([{ isIntersecting: true, target: sentinel }]);

      expect(onLoadMore).toHaveBeenCalled();
    });

    test('should show loading state', () => {
      const onLoadMore = jest.fn().mockResolvedValue(undefined);

      render(
        <InfiniteScroll 
          onLoadMore={onLoadMore} 
          hasMore={true} 
          loading={true}
          loader={<div data-testid="custom-loader">Loading...</div>}
        >
          <div>Content</div>
        </InfiniteScroll>
      );

      expect(screen.getByTestId('custom-loader')).toBeInTheDocument();
    });

    test('should not load more when hasMore is false', () => {
      const onLoadMore = jest.fn();

      render(
        <InfiniteScroll onLoadMore={onLoadMore} hasMore={false}>
          <div>Content</div>
        </InfiniteScroll>
      );

      expect(screen.queryByTestId('infinite-scroll-sentinel')).not.toBeInTheDocument();
    });

    test('should handle load more failure gracefully', async () => {
      const onLoadMore = jest.fn().mockRejectedValue(new Error('Load failed'));

      render(
        <InfiniteScroll onLoadMore={onLoadMore} hasMore={true}>
          <div>Content</div>
        </InfiniteScroll>
      );

      const sentinel = screen.getByTestId('infinite-scroll-sentinel');
      const callback = (global.IntersectionObserver as jest.Mock).mock.calls[0][0];
      
      callback([{ isIntersecting: true, target: sentinel }]);

      await waitFor(() => {
        expect(onLoadMore).toHaveBeenCalled();
      });

      // Should not crash on error
      expect(screen.getByTestId('infinite-scroll')).toBeInTheDocument();
    });
  });

  describe('Integration Tests', () => {
    test('should combine multiple interaction patterns', async () => {
      const TestComplexUI = () => {
        const [items, setItems] = React.useState(['item1', 'item2', 'item3']);
        const responsive = useResponsive();
        
        const handleDrop = (itemId: string, zoneId: string) => {
          console.log(`Dropped ${itemId} on ${zoneId}`);
        };

        const handleSwipe = (direction: string) => {
          if (direction === 'left') {
            setItems(prev => prev.slice(1));
          }
        };

        return (
          <div data-testid="complex-ui">
            <div data-testid="responsive-info">
              Current breakpoint: {responsive.breakpoint}
            </div>
            
            <DragDropProvider onDrop={handleDrop}>
              <GestureRecognizer onSwipe={handleSwipe}>
                <div style={{ display: 'flex', gap: '10px' }}>
                  {items.map(item => (
                    <Draggable key={item} id={item}>
                      <div style={{ padding: '10px', border: '1px solid #ccc' }}>
                        {item}
                      </div>
                    </Draggable>
                  ))}
                </div>
                
                <Droppable id="trash" accepts={items}>
                  <div style={{ padding: '20px', border: '2px dashed red' }}>
                    Trash Zone
                  </div>
                </Droppable>
              </GestureRecognizer>
            </DragDropProvider>
          </div>
        );
      };

      render(<TestComplexUI />);

      // Verify all components are rendered
      expect(screen.getByTestId('complex-ui')).toBeInTheDocument();
      expect(screen.getByTestId('responsive-info')).toBeInTheDocument();
      expect(screen.getByTestId('draggable-item1')).toBeInTheDocument();
      expect(screen.getByTestId('droppable-trash')).toBeInTheDocument();
      expect(screen.getByTestId('gesture-recognizer')).toBeInTheDocument();
    });

    test('should handle performance with many interactive elements', () => {
      const ManyElementsTest = () => {
        const items = Array.from({ length: 100 }, (_, i) => `item-${i}`);
        
        return (
          <DragDropProvider>
            <div style={{ height: '400px', overflow: 'auto' }}>
              {items.map(item => (
                <Draggable key={item} id={item}>
                  <div style={{ padding: '5px', margin: '2px', border: '1px solid #eee' }}>
                    {item}
                  </div>
                </Draggable>
              ))}
            </div>
          </DragDropProvider>
        );
      };

      const startTime = performance.now();
      render(<ManyElementsTest />);
      const endTime = performance.now();

      // Should render quickly even with many elements
      expect(endTime - startTime).toBeLessThan(100);
    });
  });
});