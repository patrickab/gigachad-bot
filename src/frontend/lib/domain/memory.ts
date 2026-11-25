// Shapes of the memory-store REST API (`/api/memory`): memory_stores hold directories, memories (fact lists) and documents (markdown).

/** A memory in a listing: `path` is memory_store-relative without extension, `error` marks a file the store could not read. */
export interface MemoryEntry {
  path: string
  description: string
  facts: number
  updated: string | null
  error?: "corrupt"
}

/** A document in a listing: `path` is memory_store-relative and ends in `.md`. */
export interface DocumentEntry {
  path: string
  bytes: number
  modified: string
}

/** One level of a root or memory_store: `memory_stores` are root-relative, everything else memory_store-relative. */
export interface MemoryListing {
  memory_stores: string[]
  directories: string[]
  memories: MemoryEntry[]
  documents: DocumentEntry[]
}

/** An active fact of a memory. */
export interface Fact {
  id: string
  text: string
  created: string
  updated: string
}

export type MemoryRead =
  | { kind: "memory"; description: string; facts: Fact[] }
  | { kind: "document"; text: string }

/** One write of a batch. `update` replaces a fact under a new id, `retire` hides it, `describe` sets a memory's description. */
export type MemoryOp =
  | { op: "add"; path: string; text: string; description?: string }
  | { op: "update"; path: string; id: string; text: string; description?: string }
  | { op: "retire"; path: string; id: string; description?: string }
  | { op: "describe"; path: string; description: string }
  | { op: "write_doc"; path: string; text: string }

/** Outcome of one op, in batch order; `id` is the new fact's id for add and update. */
export type MemoryOpResult =
  | { ok: true; op: MemoryOp["op"]; path: string; id?: string }
  | { ok: false; op: string | null; path: string | null; error: string; message: string }
