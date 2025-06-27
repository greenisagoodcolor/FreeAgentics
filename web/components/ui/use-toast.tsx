import { useState, useCallback } from "react";

export interface Toast {
  title: string;
  description?: string;
  duration?: number;
}

let toastId = 0;

export function useToast() {
  const [toasts, setToasts] = useState<Array<Toast & { id: number }>>([]);

  const toast = useCallback(
    ({ title, description, duration = 3000 }: Toast) => {
      const id = toastId++;
      const newToast = { id, title, description, duration };

      setToasts((prev) => [...prev, newToast]);

      // Simple console log for now since we don't have a full toast system
      console.log(`Toast: ${title}${description ? ` - ${description}` : ""}`);

      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, duration);
    },
    [],
  );

  return { toast, toasts };
}
