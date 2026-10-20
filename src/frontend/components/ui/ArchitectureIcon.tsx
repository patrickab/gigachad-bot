/** Layered-plane glyph: a gridded top layer over two plain ones, drawn like lucide icons. */
export function ArchitectureIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className}
    >
      <path d="M12 2 21 6.5 12 11 3 6.5Z" />
      <path d="M7.5 4.25 16.5 8.75M16.5 4.25 7.5 8.75" />
      <path d="M3 11l9 4.5 9-4.5" />
      <path d="M3 15.5l9 4.5 9-4.5" />
    </svg>
  )
}
