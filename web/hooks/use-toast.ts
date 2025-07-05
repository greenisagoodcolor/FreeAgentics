import * as React from "react";

export interface Toast {
  id: string;
  title?: string;
  description?: string;
  action?: React.ReactNode;
  variant?: "default" | "destructive";
}

export function useToast() {
  const [toasts, setToasts] = React.useState<Toast[]>([]);

  const toast = React.useCallback((props: Omit<Toast, "id">) => {
    const id = Math.random().toString(36).substring(2, 9);
    const newToast: Toast = { ...props, id };

    setToasts((currentToasts) => [...currentToasts, newToast]);

    // Auto remove after 5 seconds
    setTimeout(() => {
      setToasts((currentToasts) => currentToasts.filter((t) => t.id !== id));
    }, 5000);

    return { id };
  }, []);

  const dismiss = React.useCallback((toastId?: string) => {
    setToasts((currentToasts) =>
      toastId ? currentToasts.filter((t) => t.id !== toastId) : [],
    );
  }, []);

  return {
    toast,
    dismiss,
    toasts,
  };
}
