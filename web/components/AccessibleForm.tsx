import React, { useId } from "react";
import { cn } from "@/lib/utils";
import { useA11yAnnounce } from "@/lib/accessibility";

interface FormFieldProps {
  label: string;
  error?: string;
  required?: boolean;
  hint?: string;
  children: React.ReactElement;
  className?: string;
}

/**
 * Accessible form field wrapper with proper labeling and error handling
 */
export function FormField({ label, error, required, hint, children, className }: FormFieldProps) {
  const fieldId = useId();
  const errorId = `${fieldId}-error`;
  const hintId = `${fieldId}-hint`;
  const { announceError } = useA11yAnnounce();

  // Announce errors to screen readers
  React.useEffect(() => {
    if (error) {
      announceError(error);
    }
  }, [error, announceError]);

  // Clone child and add accessibility attributes
  const field = React.cloneElement(children, {
    id: fieldId,
    "aria-invalid": !!error,
    "aria-describedby": [hint && hintId, error && errorId].filter(Boolean).join(" ") || undefined,
    "aria-required": required,
  });

  return (
    <div className={cn("space-y-2", className)}>
      <label htmlFor={fieldId} className="block text-sm font-medium text-gray-700">
        {label}
        {required && (
          <span className="text-red-500 ml-1" aria-label="required">
            *
          </span>
        )}
      </label>

      {hint && (
        <p id={hintId} className="text-sm text-gray-500">
          {hint}
        </p>
      )}

      {field}

      {error && (
        <p id={errorId} className="text-sm text-red-600" role="alert" aria-live="polite">
          {error}
        </p>
      )}
    </div>
  );
}

interface RadioGroupProps {
  label: string;
  name: string;
  options: Array<{ value: string; label: string; disabled?: boolean }>;
  value?: string;
  onChange: (value: string) => void;
  error?: string;
  required?: boolean;
  className?: string;
}

/**
 * Accessible radio group with proper ARIA attributes
 */
export function RadioGroup({
  label,
  name,
  options,
  value,
  onChange,
  error,
  required,
  className,
}: RadioGroupProps) {
  const groupId = useId();
  const errorId = `${groupId}-error`;

  return (
    <fieldset
      className={cn("space-y-2", className)}
      aria-required={required}
      aria-invalid={!!error}
      aria-describedby={error ? errorId : undefined}
    >
      <legend className="text-sm font-medium text-gray-700">
        {label}
        {required && (
          <span className="text-red-500 ml-1" aria-label="required">
            *
          </span>
        )}
      </legend>

      <div className="space-y-2" role="radiogroup">
        {options.map((option) => {
          const optionId = `${groupId}-${option.value}`;

          return (
            <label
              key={option.value}
              htmlFor={optionId}
              className={cn(
                "flex items-center space-x-2 cursor-pointer",
                option.disabled && "opacity-50 cursor-not-allowed",
              )}
            >
              <input
                type="radio"
                id={optionId}
                name={name}
                value={option.value}
                checked={value === option.value}
                onChange={(e) => onChange(e.target.value)}
                disabled={option.disabled}
                className="h-4 w-4 text-blue-600 focus:ring-2 focus:ring-blue-500"
                aria-describedby={`${optionId}-label`}
              />
              <span id={`${optionId}-label`} className="text-sm text-gray-700">
                {option.label}
              </span>
            </label>
          );
        })}
      </div>

      {error && (
        <p id={errorId} className="text-sm text-red-600" role="alert" aria-live="polite">
          {error}
        </p>
      )}
    </fieldset>
  );
}

interface CheckboxGroupProps {
  label: string;
  options: Array<{
    value: string;
    label: string;
    checked: boolean;
    disabled?: boolean;
  }>;
  onChange: (value: string, checked: boolean) => void;
  error?: string;
  required?: boolean;
  className?: string;
}

/**
 * Accessible checkbox group
 */
export function CheckboxGroup({
  label,
  options,
  onChange,
  error,
  required,
  className,
}: CheckboxGroupProps) {
  const groupId = useId();
  const errorId = `${groupId}-error`;

  return (
    <fieldset
      className={cn("space-y-2", className)}
      aria-required={required}
      aria-invalid={!!error}
      aria-describedby={error ? errorId : undefined}
    >
      <legend className="text-sm font-medium text-gray-700">
        {label}
        {required && (
          <span className="text-red-500 ml-1" aria-label="required">
            *
          </span>
        )}
      </legend>

      <div className="space-y-2" role="group">
        {options.map((option) => {
          const optionId = `${groupId}-${option.value}`;

          return (
            <label
              key={option.value}
              htmlFor={optionId}
              className={cn(
                "flex items-center space-x-2 cursor-pointer",
                option.disabled && "opacity-50 cursor-not-allowed",
              )}
            >
              <input
                type="checkbox"
                id={optionId}
                value={option.value}
                checked={option.checked}
                onChange={(e) => onChange(option.value, e.target.checked)}
                disabled={option.disabled}
                className="h-4 w-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
                aria-describedby={`${optionId}-label`}
              />
              <span id={`${optionId}-label`} className="text-sm text-gray-700">
                {option.label}
              </span>
            </label>
          );
        })}
      </div>

      {error && (
        <p id={errorId} className="text-sm text-red-600" role="alert" aria-live="polite">
          {error}
        </p>
      )}
    </fieldset>
  );
}

interface FormProps extends React.FormHTMLAttributes<HTMLFormElement> {
  children: React.ReactNode;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  "aria-label"?: string;
}

/**
 * Accessible form wrapper
 */
export function Form({ children, onSubmit, "aria-label": ariaLabel, ...props }: FormProps) {
  const { announceLoading, announceSuccess, announceError } = useA11yAnnounce();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    try {
      announceLoading(true, "form submission");
      await onSubmit(e);
      announceSuccess("Form submitted successfully");
    } catch (error) {
      announceError(error instanceof Error ? error.message : "Form submission failed");
    } finally {
      announceLoading(false, "form submission");
    }
  };

  return (
    <form {...props} onSubmit={handleSubmit} noValidate aria-label={ariaLabel}>
      {children}
    </form>
  );
}

/**
 * Accessible submit button with loading state
 */
export function SubmitButton({
  children,
  isLoading,
  loadingText = "Submitting...",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  isLoading?: boolean;
  loadingText?: string;
}) {
  return (
    <button
      type="submit"
      disabled={isLoading}
      aria-busy={isLoading}
      className={cn(
        "px-4 py-2 bg-blue-600 text-white rounded-md",
        "hover:bg-blue-700 focus:outline-none focus:ring-2",
        "focus:ring-blue-500 focus:ring-offset-2",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        "transition-colors duration-200",
        className,
      )}
      {...props}
    >
      {isLoading ? (
        <>
          <span className="sr-only">{loadingText}</span>
          <span aria-hidden="true">{loadingText}</span>
        </>
      ) : (
        children
      )}
    </button>
  );
}

/**
 * Progress indicator for multi-step forms
 */
export function FormProgress({
  currentStep,
  totalSteps,
  className,
}: {
  currentStep: number;
  totalSteps: number;
  className?: string;
}) {
  const percentage = (currentStep / totalSteps) * 100;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex justify-between text-sm text-gray-600">
        <span>
          Step {currentStep} of {totalSteps}
        </span>
        <span>{Math.round(percentage)}% complete</span>
      </div>
      <div
        className="w-full bg-gray-200 rounded-full h-2"
        role="progressbar"
        aria-valuenow={currentStep}
        aria-valuemin={1}
        aria-valuemax={totalSteps}
        aria-label={`Form progress: step ${currentStep} of ${totalSteps}`}
      >
        <div
          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
