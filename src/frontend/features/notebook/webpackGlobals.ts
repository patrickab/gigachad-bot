/**
 * `@datalayer/jupyter-react` assigns webpack's `__webpack_public_path__`, which only webpack declares. Under Turbopack
 * it is an undeclared variable, and assigning one throws in a module. Imported before the library, this declares it.
 */
const scope = globalThis as { __webpack_public_path__?: string }
scope.__webpack_public_path__ ??= `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/_next/`

export {}
