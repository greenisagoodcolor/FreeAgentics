import type { Metadata } from "next";
import "./globals.css";

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
        <div className="min-h-screen bg-gray-50">{children}</div>
      </body>
    </html>
  );
}
