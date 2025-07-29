import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/components/auth-provider";
import DevAuthBootstrap from "@/components/dev-auth-bootstrap";
import SettingsModal from "@/components/SettingsModal";

// Temporarily use system fonts due to network issues
const fontClassName = "font-sans";

export const metadata: Metadata = {
  title: "FreeAgentics - Active Inference Platform",
  description: "Multi-agent system with mathematically-principled intelligence",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={fontClassName}>
        <AuthProvider>
          <DevAuthBootstrap />
          <SettingsModal />
          <div className="min-h-screen bg-white">{children}</div>
        </AuthProvider>
      </body>
    </html>
  );
}
