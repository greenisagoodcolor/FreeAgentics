"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to executive dashboard view
    router.replace("/dashboard?view=executive");
  }, [router]);

  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ background: "var(--bg-primary)" }}
    >
      <div className="text-center">
        <div
          className="w-16 h-16 mx-auto mb-4 rounded-lg flex items-center justify-center"
          style={{ background: "var(--primary-amber)" }}
        >
          <span
            className="text-2xl font-bold"
            style={{ color: "var(--bg-primary)" }}
          >
            CN
          </span>
        </div>
        <h1
          className="text-xl font-semibold mb-2"
          style={{ color: "var(--text-primary)" }}
        >
          CogniticNet
        </h1>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          Redirecting to dashboard...
        </p>
      </div>
    </div>
  );
}
