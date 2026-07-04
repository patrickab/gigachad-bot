import type { Metadata } from "next"
import { Outfit } from "next/font/google"
import { JetBrains_Mono } from "next/font/google"
import { Caveat } from "next/font/google"
import "./globals.css"

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
  weight: ["300", "400", "500", "600", "700"],
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
  weight: ["400", "500"],
})

const caveat = Caveat({
  subsets: ["latin"],
  variable: "--font-handwriting",
  display: "swap",
  weight: ["500", "600"],
})

export const metadata: Metadata = {
  title: "GigaChat Bot",
  description: "Personal AI chat interface with branching, deep research, and multi-model support",
  icons: "/assets/icon.png",
  openGraph: {
    title: "GigaChat Bot",
    description: "Personal AI chat interface",
    type: "website",
  },
}

const THEME_SCRIPT = `
  (function() {
    try {
      var stored = localStorage.getItem('theme')
      var theme = stored === 'light' ? 'light' : 'dark'
      document.documentElement.classList.toggle('light', theme === 'light')
    } catch(e) {
      document.documentElement.classList.toggle('light', false)
    }
  })()
`

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${outfit.variable} ${jetbrainsMono.variable} ${caveat.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="bg-paper text-ink antialiased">
        <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[200] focus:rounded-md focus:bg-surface focus:px-3 focus:py-1 focus:text-ink focus:text-sm">Skip to content</a>
        {children}
      </body>
    </html>
  )
}
