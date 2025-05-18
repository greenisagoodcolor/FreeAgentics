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
  return (
    <html lang="en">
      <body className={poppins.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <IsSendingProvider>
            {console.log("IsSendingProvider rendered")}
            <LLMProvider>
              {console.log("LLMProvider rendered")}
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
