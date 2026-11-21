"use client"

import dynamic from "next/dynamic"

/**
 * The app under `next dev`: rendered in the browser only.
 *
 * - With `App.dev-server.tsx` it keeps the app out of the dev server's server bundle, which then compiles the root files
 *   instead of the whole app: on every edit, and once more in full whenever the dev server restarts
 * - Less memory in use, so the dev server reaches the limit where it restarts itself (and reloads the open page) later
 * - A build never loads this file: the page it prerenders is checked by `next build`
 */
export const AppEntry = dynamic(() => import("./App").then((module) => module.App), { ssr: false })
