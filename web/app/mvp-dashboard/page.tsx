"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function MVPDashboard() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to unified dashboard with executive view (CEO-ready)
    router.replace("/dashboard?view=executive");
  }, [router]);

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-[var(--accent-primary)] mx-auto mb-4"></div>
        <h2 className="text-xl font-semibold mb-2">
          Redirecting to Executive Dashboard
        </h2>
        <p className="text-[var(--text-secondary)]">
          Loading CEO-ready interface...
        </p>
      </div>
    </div>
  );
}
