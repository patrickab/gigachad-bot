import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "Gigachad Bot",
  description: "Personal AI chat interface",
}

const THEME_SCRIPT = `
  (function() {
    try {
      var stored = localStorage.getItem('theme')
      var theme = stored === 'light' ? 'light' : 'dark'
      document.documentElement.className = theme
    } catch(e) {
      document.documentElement.className = 'dark'
    }
  })()
`

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="bg-zinc-950 text-zinc-100 antialiased">{children}</body>
    </html>
  )
}
