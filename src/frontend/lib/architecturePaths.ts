/** Any `.c4` source in a project's workspace, nested folders included: `graph/<slug>/backend/api.c4`. */
export const isArchitecturePath = (path: string) => /^graph\/[^/]+\/(?:[^/]+\/)*[^/]+\.c4$/.test(path)
/** `graph/<slug>/backend/api.c4` -> `<slug>`. */
export const architectureSlug = (path: string) => path.split("/")[1] ?? ""
/** `graph/<slug>/backend/api.c4` -> `backend/api.c4`, the path inside the workspace. */
export const architectureSource = (path: string) => path.split("/").slice(2).join("/")
