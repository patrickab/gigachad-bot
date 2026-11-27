// Loopback-only JSON API for the FastAPI backend. Stateless: every request
// carries the workspace sources and every answer is a pure function of it;
// the backend owns storage and revisions.
import { createHash } from 'node:crypto'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { readPins, type Snapshots } from './layout.ts'
import { applyOperations } from './ops/apply.ts'
import { OperationError, type Operation } from './ops/operation.ts'
import { renderModel } from './payload.ts'
import { Workspace, type Sources } from './workspace.ts'

const MAX_BODY_BYTES = 8 * 1024 * 1024

const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)

async function readBody(request: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of request) {
    size += (chunk as Buffer).length
    if (size > MAX_BODY_BYTES) throw new OperationError('Request too large')
    chunks.push(chunk as Buffer)
  }
  return Buffer.concat(chunks)
}

function parseJson(raw: Buffer): Record<string, unknown> {
  const parsed: unknown = JSON.parse(raw.toString('utf8') || '{}')
  if (!isRecord(parsed)) throw new OperationError('Expected a JSON object')
  return parsed
}

function sourcesOf(body: Record<string, unknown>): Sources {
  const sources = body.sources
  if (!isRecord(sources)) throw new OperationError('sources must be an object')
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

function sendJson(response: ServerResponse, status: number, json: string) {
  response.writeHead(status, { 'content-type': 'application/json' })
  response.end(json)
}

const send = (response: ServerResponse, status: number, payload: unknown) => sendJson(response, status, JSON.stringify(payload))

/** The last 8 `/model` answers by SHA-256 of their request body (a parse and layout costs about a second). */
const modelAnswers = new Map<string, string>()

async function modelAnswer(raw: Buffer): Promise<string> {
  const key = createHash('sha256').update(raw).digest('hex')
  let answer = modelAnswers.get(key)
  if (answer === undefined) {
    const body = parseJson(raw)
    const workspace = await Workspace.open(sourcesOf(body))
    try {
      answer = JSON.stringify(await renderModel(workspace, readPins(snapshotsOf(body))))
    } finally {
      await workspace.dispose()
    }
  }
  modelAnswers.delete(key)
  modelAnswers.set(key, answer)
  if (modelAnswers.size > 8) modelAnswers.delete(modelAnswers.keys().next().value!)
  return answer
}

async function handle(request: IncomingMessage, response: ServerResponse) {
  if (request.method === 'GET' && request.url === '/health') return send(response, 200, { status: 'ok' })
  if (request.method !== 'POST') return send(response, 405, { error: 'Method not allowed' })
  const raw = await readBody(request)
  if (request.url === '/model') return sendJson(response, 200, await modelAnswer(raw))
  const body = parseJson(raw)
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

export const server = createServer((request, response) => {
  handle(request, response).catch((error: unknown) => {
    if (error instanceof OperationError || error instanceof SyntaxError) return send(response, 422, { error: error.message })
    console.error(error)
    send(response, 500, { error: 'C4 service failed' })
  })
}).listen(port, host, () => {
  console.log(JSON.stringify({ event: 'ready', host, port }))
})
