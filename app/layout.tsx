import type React from "react"
import "./globals.css"
import { Poppins } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { LLMProvider } from "@/contexts/llm-context"
// Make sure we're explicitly importing from the .tsx file
import { IsSendingProvider } from "@/contexts/is-sending-context"
import NavBar from "@/components/nav-bar"

const poppins = Poppins({
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  console.log("RootLayout rendering")
  console.log("IsSendingProvider rendered")
  console.log("LLMProvider rendered")
  
  return (
    <html lang="en">
      <body className={poppins.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <IsSendingProvider>
            <LLMProvider>
              <NavBar />
              {children}
            </LLMProvider>
          </IsSendingProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}

export const metadata = {
      generator: 'v0.dev'
    };
