// Loopback-only JSON API for the FastAPI backend. Stateless: every request
// carries the workspace sources; the backend owns storage and revisions.
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { readPins, type Snapshots } from './layout.ts'
import { applyOperations, OperationError, type Operation } from './operations.ts'
import { Workspace, type Sources } from './workspace.ts'

const MAX_BODY_BYTES = 8 * 1024 * 1024

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of request) {
    size += (chunk as Buffer).length
    if (size > MAX_BODY_BYTES) throw new OperationError('Request too large')
    chunks.push(chunk as Buffer)
  }
  const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new OperationError('Expected a JSON object')
  return parsed as Record<string, unknown>
}

function sourcesOf(body: Record<string, unknown>): Sources {
  const sources = body.sources
  if (!sources || typeof sources !== 'object' || Array.isArray(sources)) throw new OperationError('sources must be an object')
  for (const [path, text] of Object.entries(sources)) {
    if (typeof text !== 'string' || !path.endsWith('.c4') || path.startsWith('/') || path.split('/').includes('..')) {
      throw new OperationError(`Invalid source file: ${path}`)
    }
  }
  return sources as Sources
}

/** `.likec4/<view>.likec4.snap` files: LikeC4's saved positions. */
function snapshotsOf(body: Record<string, unknown>): Snapshots {
  const snapshots = body.snapshots ?? {}
  if (typeof snapshots !== 'object' || Array.isArray(snapshots)) throw new OperationError('snapshots must be an object')
  for (const text of Object.values(snapshots)) if (typeof text !== 'string') throw new OperationError('snapshots must hold text')
  return snapshots as Snapshots
}

function send(response: ServerResponse, status: number, payload: unknown) {
  response.writeHead(status, { 'content-type': 'application/json' })
  response.end(JSON.stringify(payload))
}

async function handle(request: IncomingMessage, response: ServerResponse) {
  if (request.method === 'GET' && request.url === '/health') return send(response, 200, { status: 'ok' })
  if (request.method !== 'POST') return send(response, 405, { error: 'Method not allowed' })
  const body = await readJson(request)
  if (request.url === '/model') {
    const workspace = await Workspace.open(sourcesOf(body))
    try {
      return send(response, 200, await workspace.render(readPins(snapshotsOf(body))))
    } finally {
      await workspace.dispose()
    }
  }
  if (request.url === '/apply') {
    if (!Array.isArray(body.ops)) throw new OperationError('ops must be a list')
    const home = body.home ?? null
    if (home !== null && typeof home !== 'string') throw new OperationError('home must be a source file path')
    return send(response, 200, await applyOperations(sourcesOf(body), readPins(snapshotsOf(body)), body.ops as Operation[], home))
  }
  return send(response, 404, { error: 'Not found' })
}

const port = Number(process.env.C4_SERVICE_PORT ?? 8011)
const host = process.env.C4_SERVICE_HOST ?? '127.0.0.1'

createServer((request, response) => {
  handle(request, response).catch((error: unknown) => {
    if (error instanceof OperationError || error instanceof SyntaxError) return send(response, 422, { error: error.message })
    console.error(error)
    send(response, 500, { error: 'C4 service failed' })
  })
}).listen(port, host, () => {
  console.log(JSON.stringify({ event: 'ready', host, port }))
})
