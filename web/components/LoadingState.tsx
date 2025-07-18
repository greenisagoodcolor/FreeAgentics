// Simple LoadingState component

interface LoadingStateProps {
  message?: string;
  size?: "small" | "medium" | "large";
}

export function LoadingState({ message = "Loading...", size = "medium" }: LoadingStateProps) {
  const sizeClasses = {
    small: "w-4 h-4",
    medium: "w-8 h-8",
    large: "w-12 h-12",
  };

  return (
    <div className="flex items-center justify-center p-4">
      <div
        className={`animate-spin rounded-full border-2 border-gray-300 border-t-blue-600 ${sizeClasses[size]}`}
      ></div>
      {message && <span className="ml-2 text-gray-600">{message}</span>}
    </div>
  );
}
