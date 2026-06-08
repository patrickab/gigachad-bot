import type { Metadata } from "next"
import { Outfit } from "next/font/google"
import { JetBrains_Mono } from "next/font/google"
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

export const metadata: Metadata = {
  title: "Gigachad Bot",
  description: "Personal AI chat interface",
  icons: "/assets/icon.png",
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
    <html lang="en" suppressHydrationWarning className={`${outfit.variable} ${jetbrainsMono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="bg-paper text-ink antialiased">{children}</body>
    </html>
  )
}
