/**
 * One writer per session file (M1): an exclusive, non-blocking `flock(2)` held for a child's lifetime.
 *
 * - Consumes: libc `flock` through `bun:ffi` (Bun has no flock binding), compatible with `flock(1)` and other hosts
 * - Provides: `tryLockFile` returning a release handle, or undefined when another open file description holds it
 *
 * The descriptor is opened close-on-exec, so no child inherits another session's lock. The lock drops with
 * pi-host's process, and the children then see stdin EOF and shut down.
 */
import { dlopen, FFIType } from "bun:ffi";
import { closeSync, openSync } from "node:fs";

const LOCK_EX = 2;
const LOCK_NB = 4;

const libc = dlopen(process.platform === "darwin" ? "libSystem.B.dylib" : "libc.so.6", {
  flock: { args: [FFIType.i32, FFIType.i32], returns: FFIType.i32 },
});

export type FileLock = { release(): void };

/** Lock an existing file exclusively, undefined when it is held elsewhere. Throws ENOENT for a missing file. */
export function tryLockFile(path: string): FileLock | undefined {
  const fd = openSync(path, "r");
  if (libc.symbols.flock(fd, LOCK_EX | LOCK_NB) !== 0) {
    closeSync(fd);
    return undefined;
  }
  let released = false;
  return {
    release() {
      if (released) return;
      released = true;
      closeSync(fd);
    },
  };
}
