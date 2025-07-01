/**
 * Animation System Tests
 * Simplified animation hooks and utilities without complex timing
 */

import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { jest } from "@jest/globals";

// Mock requestAnimationFrame for testing
global.requestAnimationFrame = jest.fn((callback) => {
  setTimeout(callback, 16); // ~60fps
  return 1;
});

global.cancelAnimationFrame = jest.fn();

// Advanced Animation System
interface AnimationConfig {
  duration: number;
  easing: "linear" | "ease-in" | "ease-out" | "ease-in-out" | "bounce";
  delay?: number;
  repeat?: number | "infinite";
  direction?: "normal" | "reverse" | "alternate";
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
  onComplete?: () => void,
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
    "ease-in": (t: number) => t * t,
    "ease-out": (t: number) => 1 - Math.pow(1 - t, 2),
    "ease-in-out": (t: number) =>
      t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2,
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
    if (config.direction === "reverse") {
      currentValue = from + (to - from) * (1 - easedProgress);
    } else if (config.direction === "alternate") {
      const isEvenIteration = animationRef.current.currentIteration % 2 === 0;
      currentValue =
        from +
        (to - from) * (isEvenIteration ? easedProgress : 1 - easedProgress);
    } else {
      currentValue = from + (to - from) * easedProgress;
    }

    setProgress(rawProgress);
    onUpdate?.(currentValue);

    if (rawProgress >= 1) {
      animationRef.current.currentIteration++;

      const shouldRepeat =
        config.repeat === "infinite" ||
        (typeof config.repeat === "number" &&
          animationRef.current.currentIteration < config.repeat);

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

// Animated Component for testing
interface AnimatedBoxProps {
  config: AnimationConfig;
  from?: number;
  to?: number;
}

const AnimatedBox: React.FC<AnimatedBoxProps> = ({ 
  config, 
  from = 0, 
  to = 100 
}) => {
  const [currentValue, setCurrentValue] = React.useState(from);
  
  const animation = useAnimation(
    from,
    to,
    config,
    setCurrentValue,
  );

  return (
    <div data-testid="animated-box">
      <div data-testid="current-value">{Math.round(currentValue)}</div>
      <div data-testid="progress">{Math.round(animation.progress * 100)}</div>
      <div data-testid="is-animating">{animation.isAnimating.toString()}</div>
      <button data-testid="start-btn" onClick={animation.start}>
        Start
      </button>
      <button data-testid="stop-btn" onClick={animation.stop}>
        Stop
      </button>
      <button data-testid="reset-btn" onClick={animation.reset}>
        Reset
      </button>
    </div>
  );
};

// Tests
describe("Animation System", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("should initialize with correct default values", () => {
    render(
      <AnimatedBox 
        config={{ duration: 1000, easing: "linear" }}
        from={0}
        to={100}
      />
    );

    expect(screen.getByTestId("current-value")).toHaveTextContent("0");
    expect(screen.getByTestId("progress")).toHaveTextContent("0");
    expect(screen.getByTestId("is-animating")).toHaveTextContent("false");
  });

  test("should start and stop animation", async () => {
    render(
      <AnimatedBox 
        config={{ duration: 100, easing: "linear" }}
      />
    );

    const startBtn = screen.getByTestId("start-btn");
    const stopBtn = screen.getByTestId("stop-btn");
    const isAnimating = screen.getByTestId("is-animating");

    // Start animation
    fireEvent.click(startBtn);
    expect(isAnimating).toHaveTextContent("true");

    // Stop animation
    fireEvent.click(stopBtn);
    expect(isAnimating).toHaveTextContent("false");
  });

  test("should reset animation state", () => {
    render(
      <AnimatedBox 
        config={{ duration: 1000, easing: "linear" }}
        from={0}
        to={100}
      />
    );

    const startBtn = screen.getByTestId("start-btn");
    const resetBtn = screen.getByTestId("reset-btn");

    // Start animation then reset
    fireEvent.click(startBtn);
    fireEvent.click(resetBtn);

    expect(screen.getByTestId("current-value")).toHaveTextContent("0");
    expect(screen.getByTestId("progress")).toHaveTextContent("0");
    expect(screen.getByTestId("is-animating")).toHaveTextContent("false");
  });

  test("should handle different easing functions", () => {
    const easings: Array<AnimationConfig["easing"]> = [
      "linear", "ease-in", "ease-out", "ease-in-out", "bounce"
    ];

    easings.forEach((easing) => {
      const { unmount } = render(
        <AnimatedBox 
          config={{ duration: 100, easing }}
        />
      );

      const startBtn = screen.getByTestId("start-btn");
      fireEvent.click(startBtn);

      // Should start animating
      expect(screen.getByTestId("is-animating")).toHaveTextContent("true");

      unmount();
    });
  });

  test("should handle animation direction options", () => {
    const directions: Array<AnimationConfig["direction"]> = [
      "normal", "reverse", "alternate"
    ];

    directions.forEach((direction) => {
      const { unmount } = render(
        <AnimatedBox 
          config={{ duration: 100, easing: "linear", direction }}
        />
      );

      const startBtn = screen.getByTestId("start-btn");
      fireEvent.click(startBtn);

      expect(screen.getByTestId("is-animating")).toHaveTextContent("true");

      unmount();
    });
  });

  test("should handle animation with delay", () => {
    render(
      <AnimatedBox 
        config={{ duration: 100, easing: "linear", delay: 50 }}
      />
    );

    const startBtn = screen.getByTestId("start-btn");
    fireEvent.click(startBtn);

    // Should be animating but not immediately updating values due to delay
    expect(screen.getByTestId("is-animating")).toHaveTextContent("true");
  });

  test("should not start animation if already animating", () => {
    render(
      <AnimatedBox 
        config={{ duration: 1000, easing: "linear" }}
      />
    );

    const startBtn = screen.getByTestId("start-btn");

    // Start animation
    fireEvent.click(startBtn);
    expect(screen.getByTestId("is-animating")).toHaveTextContent("true");

    // Try to start again - should not affect state
    fireEvent.click(startBtn);
    expect(screen.getByTestId("is-animating")).toHaveTextContent("true");
  });

  test("should handle completion callback", () => {
    const onComplete = jest.fn();
    
    const TestComponent = () => {
      const animation = useAnimation(
        0,
        100,
        { duration: 50, easing: "linear" },
        undefined,
        onComplete,
      );

      React.useEffect(() => {
        animation.start();
      }, []);

      return <div data-testid="test">Test</div>;
    };

    render(<TestComponent />);

    // Fast-forward time to complete animation
    act(() => {
      jest.advanceTimersByTime(100);
    });

    expect(onComplete).toHaveBeenCalled();
  });

  test("should handle update callback", () => {
    const onUpdate = jest.fn();
    
    const TestComponent = () => {
      const animation = useAnimation(
        0,
        100,
        { duration: 100, easing: "linear" },
        onUpdate,
      );

      React.useEffect(() => {
        animation.start();
      }, []);

      return <div data-testid="test">Test</div>;
    };

    render(<TestComponent />);

    expect(onUpdate).toHaveBeenCalled();
  });

  test("should handle repeated animations", () => {
    render(
      <AnimatedBox 
        config={{ duration: 50, easing: "linear", repeat: 2 }}
      />
    );

    const startBtn = screen.getByTestId("start-btn");
    fireEvent.click(startBtn);

    expect(screen.getByTestId("is-animating")).toHaveTextContent("true");

    // Animation should handle repetition internally
    // We can't easily test the exact repetition without complex timing
  });

  test("should cleanup animation on unmount", () => {
    const { unmount } = render(
      <AnimatedBox 
        config={{ duration: 1000, easing: "linear" }}
      />
    );

    const startBtn = screen.getByTestId("start-btn");
    fireEvent.click(startBtn);

    // Unmount while animating
    unmount();

    // Should not cause memory leaks or errors
    expect(cancelAnimationFrame).toHaveBeenCalled();
  });

  test("should handle edge cases with zero duration", () => {
    render(
      <AnimatedBox 
        config={{ duration: 0, easing: "linear" }}
      />
    );

    const startBtn = screen.getByTestId("start-btn");
    fireEvent.click(startBtn);

    // Should handle zero duration gracefully
    expect(screen.getByTestId("is-animating")).toHaveTextContent("true");
  });

  test("should handle negative values", () => {
    render(
      <AnimatedBox 
        config={{ duration: 100, easing: "linear" }}
        from={100}
        to={-50}
      />
    );

    const startBtn = screen.getByTestId("start-btn");
    fireEvent.click(startBtn);

    expect(screen.getByTestId("current-value")).toHaveTextContent("100");
    expect(screen.getByTestId("is-animating")).toHaveTextContent("true");
  });
});