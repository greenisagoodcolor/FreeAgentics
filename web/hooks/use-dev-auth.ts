"use client";

import { useEffect } from "react";

export default function useDevAuth() {
  useEffect(() => {
    // In dev mode, we get the auth token from the backend automatically
    if (process.env.NODE_ENV === "development") {
      // The auth is handled by the backend dev_auth middleware
      // No client-side token fetching needed in the new architecture
      console.log("Dev auth enabled - authentication handled by backend");
    }
  }, []);
}