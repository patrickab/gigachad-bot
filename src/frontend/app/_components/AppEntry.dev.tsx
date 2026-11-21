"use client"

import dynamic from "next/dynamic"

/**
 * The app under `next dev`: rendered in the browser only.
 *
 * - The dev server compiles the app once, for the browser, not a second time for a server render: less work on every
 *   edit, less memory
 * - A build never loads this file: the page it prerenders is checked by `next build`
 */
export const AppEntry = dynamic(() => import("./App").then((module) => module.App), { ssr: false })
