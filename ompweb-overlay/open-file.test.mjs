import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, { tsconfigPaths: true });
const { resolveOpenFilePath } = await jiti.import("./embed/open-file.ts");

test("relative tool paths resolve against the session cwd", () => {
  assert.equal(resolveOpenFilePath("src/a.ts", "/home/me/project"), "/home/me/project/src/a.ts");
  assert.equal(resolveOpenFilePath("./src/a.ts:12", "/home/me/project"), "/home/me/project/src/a.ts");
});

test("absolute paths pass through and an unresolvable path is kept", () => {
  assert.equal(resolveOpenFilePath("/etc/hosts", "/home/me/project"), "/etc/hosts");
  assert.equal(resolveOpenFilePath("src/a.ts", undefined), "src/a.ts");
  assert.equal(resolveOpenFilePath("../outside.ts", "/home/me/project"), "../outside.ts");
});
