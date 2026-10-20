// Digest of everything that goes into the packed tarball, recorded in SOURCE.sha256 as "<version> <sha256>".
// Unlike the tarball bytes it does not depend on the npm or zlib version, so a test can check it without the fork.
import { createHash } from "node:crypto"
import { existsSync, readdirSync, readFileSync } from "node:fs"
import { join, relative } from "node:path"

export const RECORD_FILE = "SOURCE.sha256"

/** Hash of package.json, tsconfig.build.json and every file under src/, by relative name and content. */
export function sourceDigest(packageDir) {
  const sources = readdirSync(join(packageDir, "src"), { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => relative(packageDir, join(entry.parentPath, entry.name)))
  const hash = createHash("sha256")
  for (const file of ["package.json", "tsconfig.build.json", ...sources].sort()) {
    hash.update(`${file}\0`).update(readFileSync(join(packageDir, file))).update("\0")
  }
  return hash.digest("hex")
}

/** The recorded `{ version, digest }`, or null before the first sync. */
export function readRecord(packageDir) {
  const file = join(packageDir, RECORD_FILE)
  if (!existsSync(file)) return null
  const [version, digest] = readFileSync(file, "utf8").trim().split(" ")
  return { version, digest }
}

/** Whether the sources changed since the record while `version` stayed the same, which a consumer's lock cannot survive. */
export function isChangedWithoutVersionBump(packageDir, version) {
  const record = readRecord(packageDir)
  return record !== null && record.version === version && record.digest !== sourceDigest(packageDir)
}
