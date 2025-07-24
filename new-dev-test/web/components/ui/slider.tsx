import * as React from "react";

export interface SliderProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "value"> {
  value?: number[];
  onValueChange?: (value: number[]) => void;
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ className, value, min = 0, max = 100, step = 1, onValueChange, onChange, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = Number(e.target.value);
      onChange?.(e);
      onValueChange?.([newValue]);
    };

    const currentValue = Array.isArray(value) ? value[0] : value || 0;
    const percentage = ((Number(currentValue) - Number(min)) / (Number(max) - Number(min))) * 100;

    return (
      <div
        className={`relative flex w-full touch-none select-none items-center ${className || ""}`}
      >
        <div className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
          <div className="absolute h-full bg-primary" style={{ width: `${percentage}%` }} />
        </div>
        <input
          type="range"
          ref={ref}
          min={min}
          max={max}
          step={step}
          value={currentValue}
          onChange={handleChange}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
          {...props}
        />
      </div>
    );
  },
);
Slider.displayName = "Slider";

export { Slider };
