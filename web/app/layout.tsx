import type React from "react";
import "../globals.css";
import "../styles/design-tokens.css";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/themeprovider";
import { LLMProvider } from "@/contexts/llm-context";
import { IsSendingProvider } from "@/contexts/is-sending-context";
import { ReduxProvider } from "@/providers/ReduxProvider";
import NavBar from "@/components/navbar";

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
        className={`${inter.variable} ${jetbrainsMono.variable} dark bg-[#0A0A0B] text-white`}
      >
        <ReduxProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem={false}
            disableTransitionOnChange
          >
            <IsSendingProvider>
              <LLMProvider>
                <NavBar />
                <main className="pt-16">{children}</main>
              </LLMProvider>
            </IsSendingProvider>
          </ThemeProvider>
        </ReduxProvider>
      </body>
    </html>
  );
}

export const metadata = {
  title: "FreeAgentics - Multi-Agent AI System",
  description:
    "AI agents with Active Inference minds forming coalitions and businesses",
  generator: "Next.js",
};
