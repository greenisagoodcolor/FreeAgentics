import { useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

export interface User {
  id: string;
  email: string;
  name: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AUTH_TOKEN_KEY = "fa.jwt";
const USER_KEY = "fa.user";

// Helper function to check if JWT token is expired
function isTokenExpired(token: string): boolean {
  // Handle dev tokens that might not be JWTs
  if (token === "dev" || token.startsWith("dev_")) {
    console.log("✅ Dev token detected, never expires");
    return false;
  }

  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    const now = Math.floor(Date.now() / 1000);
    const isExpired = payload.exp < now;
    if (isExpired) {
      console.log("🔄 JWT token expired, needs refresh");
    }
    return isExpired;
  } catch (error) {
    console.warn("⚠️ Token validation failed, treating as expired:", error);
    return true; // If we can't parse it, consider it expired
  }
}

export function useAuth(): AuthState {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  
  // Prevent infinite loops and race conditions
  const isInitializingRef = useRef(false);
  const lastFetchAttemptRef = useRef<number>(0);
  const FETCH_DEBOUNCE_MS = 1000; // Prevent rapid successive fetches

  // Load auth state from localStorage and fetch dev token if needed
  useEffect(() => {
    const loadAuthState = async () => {
      // Prevent multiple simultaneous initialization attempts
      if (isInitializingRef.current) {
        console.log("[Auth] Initialization already in progress, skipping...");
        return;
      }
      
      // Debounce rapid successive calls
      const now = Date.now();
      if (now - lastFetchAttemptRef.current < FETCH_DEBOUNCE_MS) {
        console.log("[Auth] Debouncing rapid auth initialization");
        return;
      }
      
      isInitializingRef.current = true;
      lastFetchAttemptRef.current = now;
      
      try {
        const storedToken = localStorage.getItem(AUTH_TOKEN_KEY);
        const storedUser = localStorage.getItem(USER_KEY);

        // Check if stored token exists and is not expired
        if (storedToken && storedUser && !isTokenExpired(storedToken)) {
          console.log("✅ Using valid cached token");
          setToken(storedToken);
          setUser(JSON.parse(storedUser));
          setIsLoading(false);
          return;
        }

        // Clear expired/invalid tokens
        if (storedToken) {
          console.log("🔄 Clearing expired token from localStorage");
          localStorage.removeItem(AUTH_TOKEN_KEY);
          localStorage.removeItem(USER_KEY);
        }

        // DevToken bootstrap for dev mode - only attempt once per session
        console.log("[Auth] No valid token in storage, fetching dev token...");
        try {
          const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
          console.log(`[Auth] Fetching dev config from: ${backendUrl}/api/v1/dev-config`);
          const response = await fetch(`${backendUrl}/api/v1/dev-config`);

          if (!response.ok) {
            throw new Error(`Dev config request failed: ${response.status}`);
          }

          const config = await response.json();
          const token = config.auth?.token;

          if (token) {
            console.log("[Auth] Received dev token, storing...");
            localStorage.setItem(AUTH_TOKEN_KEY, token);
            const devUser = {
              id: "dev-user",
              email: "developer@freeagentics.dev",
              name: "Developer",
            };
            localStorage.setItem(USER_KEY, JSON.stringify(devUser));
            setToken(token);
            setUser(devUser);
            console.log("[Auth] ✅ Dev token loaded and auth state updated");
          } else if (config.mode === "dev" && !config.features?.auth_required) {
            // In dev mode without auth, use a simple "dev" token
            console.log("[Auth] Dev mode without auth, using simple dev token");
            const devToken = "dev";
            localStorage.setItem(AUTH_TOKEN_KEY, devToken);
            const devUser = {
              id: "dev-user",
              email: "developer@freeagentics.dev",
              name: "Developer",
            };
            localStorage.setItem(USER_KEY, JSON.stringify(devUser));
            setToken(devToken);
            setUser(devUser);
            console.log("[Auth] ✅ Simple dev token set for auth-free dev mode");
          } else {
            console.warn("[Auth] No token found in dev config response");
          }
        } catch (devConfigError) {
          console.error("[Auth] Dev config error:", devConfigError);
          console.log("[Auth] Continuing without auto-auth");
        }
      } catch (error) {
        console.error("Failed to load auth state:", error);
      } finally {
        setIsLoading(false);
        isInitializingRef.current = false;
      }
    };

    loadAuthState();
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    setIsLoading(true);
    try {
      // TODO: Replace with actual API call
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error("Login failed");
      }

      const data = await response.json();
      const { token, user: userData } = data;

      // Store auth data
      localStorage.setItem(AUTH_TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(userData));

      setToken(token);
      setUser(userData);
    } catch (error) {
      console.error("Login error:", error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setIsLoading(true);
    try {
      // TODO: Call logout API endpoint if needed
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${localStorage.getItem(AUTH_TOKEN_KEY)}`,
        },
      });
    } catch (error) {
      console.error("Logout error:", error);
    } finally {
      // Clear auth data regardless of API call result
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      setToken(null);
      setUser(null);
      setIsLoading(false);

      // Redirect to login page
      router.push("/login");
    }
  }, [router]);

  return {
    user,
    isAuthenticated: !!user,
    isLoading,
    token,
    login,
    logout,
  };
}
