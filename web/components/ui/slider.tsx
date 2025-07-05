import * as React from "react";

export interface SliderProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number[];
  onValueChange?: (value: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
}

const Slider = React.forwardRef<HTMLDivElement, SliderProps>(
  (
    {
      className,
      value: _value,
      onValueChange: _onValueChange,
      min: _min,
      max: _max,
      step: _step,
      ...props
    },
    ref,
  ) => (
    <div
      ref={ref}
      className={`relative flex w-full touch-none select-none items-center ${className || ""}`}
      {...props}
    >
      <div className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
        <div className="absolute h-full bg-primary" style={{ width: "50%" }} />
      </div>
      <div className="block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
    </div>
  ),
);
Slider.displayName = "Slider";

export { Slider };
