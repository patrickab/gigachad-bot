import type { Metadata } from "next"
import "./globals.css"

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
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="bg-paper text-ink antialiased">{children}</body>
    </html>
  )
}
