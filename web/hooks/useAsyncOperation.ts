import { useState, useEffect, useCallback } from "react";

interface UseAsyncOperationResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  reset: () => void;
  execute: () => void;
}

export function useAsyncOperation<T>(
  asyncFunction: () => Promise<T>,
): UseAsyncOperationResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const execute = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await asyncFunction();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [asyncFunction]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  useEffect(() => {
    execute();
  }, [execute]);

  return {
    data,
    loading,
    error,
    reset,
    execute,
  };
}
