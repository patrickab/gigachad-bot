import { defineConfig } from "vitest/config"
import react from "@vitejs/plugin-react"
import path from "node:path"

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["**/__tests__/**/*.test.{ts,tsx}"],
    exclude: ["node_modules", ".next"],
  },
  resolve: {
    alias: [
      { find: /^@gigachad\/chat-ui$/, replacement: path.resolve(__dirname, "packages/chat-ui/src/index.ts") },
      { find: /^@gigachad\/chat-ui\//, replacement: `${path.resolve(__dirname, "packages/chat-ui/src")}/` },
      { find: /^@\//, replacement: `${path.resolve(__dirname)}/` },
    ],
  },
})
