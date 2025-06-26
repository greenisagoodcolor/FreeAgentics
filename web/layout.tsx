import type React from "react";
import "./globals.css";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/themeprovider";
import { LLMProvider } from "@/contexts/llm-context";
// Make sure we're explicitly importing from the .tsx file
import { IsSendingProvider } from "@/contexts/is-sending-context";
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
  console.log("RootLayout rendering");
  console.log("IsSendingProvider rendered");
  console.log("LLMProvider rendered");

  return (
    <html lang="en">
      <body className={`${inter.variable} ${jetbrainsMono.variable}`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <IsSendingProvider>
            <LLMProvider>
              <NavBar />
              {children}
            </LLMProvider>
          </IsSendingProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}

export const metadata = {
  generator: "v0.dev",
};
