import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { readRecord, sourceDigest } from "../source-hash.mjs"

const packageDir = join(dirname(fileURLToPath(import.meta.url)), "..")
const { version } = (await import("../package.json")).default

describe("@gigachad/chat-ui source record", () => {
  it("matches the sources, else the fork still has an older package", () => {
    expect(readRecord(packageDir), "SOURCE.sha256 missing: run npm run sync:ompweb").toEqual({ version, digest: sourceDigest(packageDir) })
  })
})
