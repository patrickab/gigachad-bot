/**
 * `App.tsx` in the server bundle of `next dev` (`next.config.mjs`).
 *
 * Never rendered: `AppEntry.dev.tsx` loads the app in the browser only. It keeps the app's modules out of the server
 * bundle, which would build every one of them although it runs none.
 */
export function App() {
  return null
}
