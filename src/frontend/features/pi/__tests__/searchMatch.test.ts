import { describe, expect, it } from "vitest"
import { compileSearchQuery as hostCompile, findSearchMatches as hostFind } from "../../../../../pi-host/src/host/local/query"
import { compileSearchQuery, findSearchMatches } from "../searchMatch"

const TEXTS = [
  "The frobnicator lives in settings; frobnicators, Frobnicater and a frob.",
  "Naïve café résumé, NAÏVE CAFÉ, 日本語のテキスト and ‹quotes›",
  "run the tests, then run the build; build builds banana bananas",
  "",
]
const QUERIES = ["frobnicator", "frob settings", "\"the build\"", "\"run the\" tests", "banana", "naïve", "日本語", "a", "frobnicater", "x y z", "\"unclosed"]

describe("the browser's copy of the host's search matching", () => {
  it("finds the same spans as pi-host for every query and text", () => {
    for (const query of QUERIES) {
      const ours = compileSearchQuery(query)
      const theirs = hostCompile(query)
      expect(Boolean(ours)).toBe(Boolean(theirs))
      if (!ours || !theirs) continue
      for (const text of TEXTS) expect(findSearchMatches(text, ours), `${query} in ${text}`).toEqual(hostFind(text, theirs))
    }
  })

  it("has no query for blank text", () => {
    expect(compileSearchQuery("  ")).toBeNull()
  })
})
