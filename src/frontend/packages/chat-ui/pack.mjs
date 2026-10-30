#!/usr/bin/env node
// Builds @gigachad/chat-ui from src/ and packs it as <out>/gigachad-chat-ui-<version>.tgz.
//
//   node packages/chat-ui/pack.mjs --out <dir>
//
// Run from src/frontend, which supplies tsc and the dependency types. The package source here is the only copy:
// the shell imports it as `@/packages/chat-ui/src`, run-omp-web.sh packs it on every start and installs the
// tarball into the served omp-web checkout for the overlay (`ompweb-overlay/`).
//
// Sources import siblings with their .ts/.tsx extension so the shell's bundler, vitest and Node-valid output all
// resolve them. The tarball is reproducible: tsc output is deterministic and npm pack writes fixed timestamps,
// so its digest changes only with its sources.
import { execFileSync } from "node:child_process"
import { copyFileSync, cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { createRequire } from "node:module"
import { tmpdir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const packageDir = dirname(fileURLToPath(import.meta.url))
const args = process.argv.slice(2)
const outIndex = args.indexOf("--out")
if (outIndex === -1 || !args[outIndex + 1]) throw new Error("usage: node packages/chat-ui/pack.mjs --out <dir>")
const outDir = resolve(args[outIndex + 1])

const manifest = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"))
const tarballName = `${manifest.name.replace(/^@/, "").replace("/", "-")}-${manifest.version}.tgz`
const tsc = createRequire(join(packageDir, "pack.mjs")).resolve("typescript/bin/tsc")

const staging = mkdtempSync(join(tmpdir(), "gigachad-chat-ui-"))
try {
  execFileSync(process.execPath, [tsc, "-p", join(packageDir, "tsconfig.build.json"), "--outDir", join(staging, "dist")], { stdio: "inherit" })
  // tsc rewrites `./x.tsx` to `./x.js` in the emitted JS but not in the declarations, which a consumer could not resolve.
  for (const file of readdirSync(join(staging, "dist")).filter((name) => name.endsWith(".d.ts"))) {
    const path = join(staging, "dist", file)
    writeFileSync(path, readFileSync(path, "utf8").replace(/(from "\.\/[^"]+)\.tsx?"/g, '$1.js"'))
  }
  // Stylesheets ship as-is, `@gigachad/chat-ui/<name>.css`. The shell imports the same files from src/.
  for (const stylesheet of readdirSync(join(packageDir, "src")).filter((name) => name.endsWith(".css"))) {
    copyFileSync(join(packageDir, "src", stylesheet), join(staging, "dist", stylesheet))
  }
  // The font files `fonts.css` points at, next to it as in src/.
  cpSync(join(packageDir, "src", "fonts"), join(staging, "dist", "fonts"), { recursive: true })
  copyFileSync(join(packageDir, "package.json"), join(staging, "package.json"))
  mkdirSync(outDir, { recursive: true })
  execFileSync("npm", ["pack", "--ignore-scripts", "--silent", "--pack-destination", outDir], { cwd: staging, stdio: ["ignore", "ignore", "inherit"] })
  console.log(join(outDir, tarballName))
} finally {
  rmSync(staging, { recursive: true, force: true })
}
