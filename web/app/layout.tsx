import type React from "react";
import "../styles/globals.css";
import "../styles/design-tokens.css";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/themeprovider";
import { LLMProvider } from "@/contexts/llm-context";
import { IsSendingProvider } from "@/contexts/is-sending-context";
import { ReduxProvider } from "@/providers/ReduxProvider";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} dark bg-[var(--bg-primary)] text-[var(--text-primary)]`}
        style={{ fontFamily: "var(--font-primary)" }}
      >
        <ReduxProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem={false}
            disableTransitionOnChange
          >
            <IsSendingProvider>
              <LLMProvider>{children}</LLMProvider>
            </IsSendingProvider>
          </ThemeProvider>
        </ReduxProvider>
      </body>
    </html>
  );
}

export const metadata = {
  title: "FreeAgentics - Multi-Agent AI Dashboard",
  description:
    "Professional Bloomberg-style dashboard for multi-agent AI systems",
  generator: "Next.js",
};
