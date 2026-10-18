import type { Metadata } from "next"
import { Outfit } from "next/font/google"
import { JetBrains_Mono } from "next/font/google"
import { Architects_Daughter } from "next/font/google"
import localFont from "next/font/local"
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

const architectsDaughter = Architects_Daughter({
  subsets: ["latin"],
  variable: "--font-handwriting",
  display: "swap",
  weight: ["400"],
})

// Nerd Font icons (omp session titles start with one). PUA only, fetched when a page shows such a glyph.
const symbolsNerdFont = localFont({
  src: "./fonts/SymbolsNerdFontMono-Regular.woff2",
  variable: "--font-symbols",
  display: "swap",
  preload: false,
  adjustFontFallback: false,
  declarations: [{ prop: "unicode-range", value: "U+E000-F8FF, U+F0000-FFFFD" }],
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

// Runs before hydration so the saved theme applies without a flash. Default is dark.
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

// Suppress the long-press context menu for touch pointers only, mouse right-click stays.
const NO_TOUCH_CONTEXTMENU_SCRIPT = `
  (function() {
    var touchActive = false
    window.addEventListener('pointerdown', function(e) { touchActive = e.pointerType === 'touch' }, true)
    window.addEventListener('pointerup', function() { touchActive = false }, true)
    window.addEventListener('contextmenu', function(e) { if (touchActive) e.preventDefault() }, true)
  })()
`

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${outfit.variable} ${jetbrainsMono.variable} ${architectsDaughter.variable} ${symbolsNerdFont.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
        <script dangerouslySetInnerHTML={{ __html: NO_TOUCH_CONTEXTMENU_SCRIPT }} />
      </head>
      <body spellCheck={false} className="bg-paper text-ink antialiased">
        <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[200] focus:rounded-md focus:bg-surface focus:px-3 focus:py-1 focus:text-ink focus:text-sm">Skip to content</a>
        {children}
      </body>
    </html>
  )
}
