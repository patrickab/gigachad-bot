import { readdirSync, readFileSync, statSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"

// The tarball LearnHarness installs holds only what package.json `exports` names; these checks stand in for its build.
const packageDir = join(import.meta.dirname, "..")
const overlayDir = join(packageDir, "..", "..", "..", "..", "learnharness-overlay")
const read = (path: string) => readFileSync(path, "utf8")
const manifest = JSON.parse(read(join(packageDir, "package.json")))
const exportKeys = Object.keys(manifest.exports)
const stylesheets = exportKeys.filter((key) => key.endsWith(".css")).map((key) => key.slice(2))
const modules = exportKeys.filter((key) => !key.endsWith(".css")).map((key) => `src/${key.slice(2)}.ts`)

function filesUnder(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    return statSync(path).isDirectory() ? filesUnder(path) : [path]
  })
}

describe("the shipped chat-ui package", () => {
  it("exports every specifier the overlay imports", () => {
    const specifiers = new Set(
      filesUnder(overlayDir).flatMap((file) => [...read(file).matchAll(/["']@gigachad\/chat-ui(\/[^"']+)?["']/g)].map((match) => match[1] ?? "")),
    )
    expect(specifiers.size).toBeGreaterThan(0)
    for (const specifier of specifiers) expect(exportKeys).toContain(`.${specifier}`)
  })

  it("builds every export from its own source", () => {
    for (const name of stylesheets) expect(statSync(join(packageDir, "src", name)).isFile()).toBe(true)
    for (const path of modules) expect(statSync(join(packageDir, path)).isFile()).toBe(true)
    expect(JSON.parse(read(join(packageDir, "tsconfig.build.json"))).include).toEqual(modules)
  })

  it("declares no dependencies and no root entry", () => {
    for (const field of ["dependencies", "peerDependencies", "main"]) expect(manifest).not.toHaveProperty([field])
    expect(manifest.exports).not.toHaveProperty(["."])
  })

  it("ships files that need nothing outside the tarball", () => {
    for (const path of modules) expect(read(join(packageDir, path))).not.toMatch(/^\s*(import\b|export\b.*\bfrom\b)/m)
    for (const name of stylesheets) {
      const css = read(join(packageDir, "src", name))
      expect(css).not.toContain("@import")
      for (const match of css.matchAll(/url\(\s*["']?([^"')]+)/g)) expect(match[1]).toMatch(/^\.\/fonts\//)
    }
  })
})
