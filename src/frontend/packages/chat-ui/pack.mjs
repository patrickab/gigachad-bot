#!/usr/bin/env node
// Builds @gigachad/chat-ui from src/ and packs it as gigachad-chat-ui-<version>.tgz.
//
//   node packages/chat-ui/pack.mjs --out <dir> --app <app> [--check | --install]
//   npm run sync:ompweb    packs into the ompweb fork and installs it there (GIGACHAD_OMP_WEB_REPO, default ~/git/ompweb)
//   npm run check:ompweb   verifies the fork carries the current tarball, manifest version and lock integrity
//
// Run from src/frontend, which supplies tsc and the dependency types. Sources import siblings with their
// .ts/.tsx extension so the shell's bundler, vitest and Node-valid output all resolve them. The package source here is the only
// copy: the shell imports it as `@/packages/chat-ui/src`, other apps (the ompweb fork) install the tarball.
//
// - --out <dir>      where the tarball goes, required
// - --app <app>      the consuming app, required for --check and --install
// - --check          packs to a temp dir and fails when <dir>'s tarball differs, or when <app>'s manifest version or lock
//                    integrity do not match the packed tarball
// - --install        then removes older tarballs from <dir> and `npm install`s this one in <app>, updating its manifest and lock
//
// Every run records the sources' digest in SOURCE.sha256 and refuses to pack changed sources under an unchanged version.
// A test checks the record without the fork, so a source change without `npm run sync:ompweb` fails in the shell.
//
// The tarball is reproducible: tsc output is deterministic and npm pack writes fixed timestamps.
// Bump "version" in package.json for every source change that ships to a consumer, a consumer's lock
// pins the tarball's integrity hash, so a changed tarball under an unchanged version breaks its `npm ci`.
import { execFileSync } from "node:child_process"
import { createHash } from "node:crypto"
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs"
import { createRequire } from "node:module"
import { tmpdir } from "node:os"
import { dirname, join, relative, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { RECORD_FILE, isChangedWithoutVersionBump, sourceDigest } from "./source-hash.mjs"

const packageDir = dirname(fileURLToPath(import.meta.url))
const args = process.argv.slice(2)
const check = args.includes("--check")
const outIndex = args.indexOf("--out")
const install = args.includes("--install")
const appIndex = args.indexOf("--app")
const usage = "usage: node packages/chat-ui/pack.mjs --out <dir> --app <app> [--check | --install]"
if (outIndex === -1 || !args[outIndex + 1]) throw new Error(usage)
if ((check || install) && !args[appIndex + 1]) throw new Error(usage)
const outDir = resolve(args[outIndex + 1])
const appDir = appIndex === -1 ? null : resolve(args[appIndex + 1])

const manifest = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"))
const tarballName = `${manifest.name.replace(/^@/, "").replace("/", "-")}-${manifest.version}.tgz`
const tsc = createRequire(join(packageDir, "pack.mjs")).resolve("typescript/bin/tsc")

if (!check && isChangedWithoutVersionBump(packageDir, manifest.version)) {
  console.error(`packages/chat-ui sources changed since ${manifest.version} was packed, bump "version" in packages/chat-ui/package.json, then run: npm run sync:ompweb`)
  process.exit(1)
}

/** What differs between the packed tarball and the consuming app's vendored tarball, manifest and lock. */
function checkConsumer(packed) {
  const problems = []
  const vendored = join(outDir, tarballName)
  if (!existsSync(vendored)) problems.push(`${vendored} is missing`)
  else if (!readFileSync(vendored).equals(packed)) problems.push(`${vendored} differs from the freshly packed tarball`)
  const spec = `file:${relative(appDir, vendored)}`
  const appManifest = JSON.parse(readFileSync(join(appDir, "package.json"), "utf8"))
  if (appManifest.dependencies?.[manifest.name] !== spec) problems.push(`${join(appDir, "package.json")} depends on ${appManifest.dependencies?.[manifest.name]}, expected ${spec}`)
  const locked = JSON.parse(readFileSync(join(appDir, "package-lock.json"), "utf8")).packages?.[`node_modules/${manifest.name}`]
  const integrity = `sha512-${createHash("sha512").update(packed).digest("base64")}`
  if (locked?.version !== manifest.version) problems.push(`package-lock.json pins ${manifest.name} ${locked?.version}, expected ${manifest.version}`)
  if (locked?.integrity !== integrity) problems.push("package-lock.json integrity does not match the packed tarball, the fork's `npm ci` would fail")
  return problems
}

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
  copyFileSync(join(packageDir, "package.json"), join(staging, "package.json"))
  const destination = check ? join(staging, "check") : outDir
  mkdirSync(destination, { recursive: true })
  execFileSync("npm", ["pack", "--ignore-scripts", "--silent", "--pack-destination", destination], { cwd: staging, stdio: ["ignore", "ignore", "inherit"] })
  const packed = readFileSync(join(destination, tarballName))
  const digest = createHash("sha256").update(packed).digest("hex")
  if (check) {
    const problems = checkConsumer(packed)
    if (problems.length > 0) {
      for (const problem of problems) console.error(problem)
      console.error("Fix: bump \"version\" in packages/chat-ui/package.json (unless only the fork is behind), then run: npm run sync:ompweb")
      process.exit(1)
    }
    console.log(`${tarballName} is up to date in ${appDir} (sha256 ${digest})`)
  } else {
    console.log(`${join(outDir, tarballName)} sha256 ${digest}`)
    writeFileSync(join(packageDir, RECORD_FILE), `${manifest.version} ${sourceDigest(packageDir)}\n`)
  }
  if (install) {
    const packagePrefix = tarballName.replace(/-[^-]+\.tgz$/, "-")
    for (const stale of readdirSync(outDir).filter((name) => name.startsWith(packagePrefix) && name.endsWith(".tgz") && name !== tarballName)) {
      unlinkSync(join(outDir, stale))
    }
    execFileSync("npm", ["install", "--save", "--ignore-scripts", `./${relative(appDir, join(outDir, tarballName))}`], { cwd: appDir, stdio: "inherit" })
  }
} finally {
  rmSync(staging, { recursive: true, force: true })
}
