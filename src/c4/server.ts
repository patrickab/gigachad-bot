// Loopback-only JSON API for the FastAPI backend. Stateless: every request
// carries the workspace sources and every answer is a pure function of it;
// the backend owns storage and revisions.
import { createHash } from 'node:crypto'
import { realpathSync } from 'node:fs'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { fileURLToPath } from 'node:url'
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
  if (!isRecord(snapshots)) throw new OperationError('snapshots must be an object')
  for (const text of Object.values(snapshots)) if (typeof text !== 'string') throw new OperationError('snapshots must hold text')
  return snapshots as Snapshots
}

/** Writes `payload` as JSON; a string is already JSON. */
function send(response: ServerResponse, status: number, payload: unknown) {
  response.writeHead(status, { 'content-type': 'application/json' })
  response.end(typeof payload === 'string' ? payload : JSON.stringify(payload))
}

/** `/model` answers by SHA-256 of their request body (a parse and layout costs about a second): at most 8 and 32 MB, oldest evicted first; identical concurrent requests share one run. */
const MODEL_MEMO_BYTES = 32 * 1024 * 1024
const modelAnswers = new Map<string, string>()
const modelRuns = new Map<string, Promise<string>>()

async function renderAnswer(raw: Buffer): Promise<string> {
  const body = parseJson(raw)
  const workspace = await Workspace.open(sourcesOf(body))
  try {
    return JSON.stringify(await renderModel(workspace, readPins(snapshotsOf(body))))
  } finally {
    await workspace.dispose()
  }
}

async function modelAnswer(raw: Buffer): Promise<string> {
  const key = createHash('sha256').update(raw).digest('hex')
  let answer = modelAnswers.get(key)
  if (answer === undefined) {
    let run = modelRuns.get(key)
    if (!run) modelRuns.set(key, run = renderAnswer(raw).finally(() => modelRuns.delete(key)))
    answer = await run
  }
  modelAnswers.delete(key)
  modelAnswers.set(key, answer)
  let bytes = 0
  for (const text of modelAnswers.values()) bytes += text.length
  for (const oldest of modelAnswers.keys()) {
    if (modelAnswers.size <= 8 && bytes <= MODEL_MEMO_BYTES || oldest === key) break
    bytes -= modelAnswers.get(oldest)!.length
    modelAnswers.delete(oldest)
  }
  return answer
}

async function handle(request: IncomingMessage, response: ServerResponse) {
  if (request.method === 'GET' && request.url === '/health') return send(response, 200, { status: 'ok' })
  if (request.method !== 'POST') return send(response, 405, { error: 'Method not allowed' })
  const raw = await readBody(request)
  if (request.url === '/model') return send(response, 200, await modelAnswer(raw))
  const body = parseJson(raw)
  if (request.url === '/apply') {
    if (!Array.isArray(body.ops)) throw new OperationError('ops must be a list')
    const home = body.home ?? null
    if (home !== null && typeof home !== 'string') throw new OperationError('home must be a source file path')
    return send(response, 200, await applyOperations(sourcesOf(body), readPins(snapshotsOf(body)), body.ops as Operation[], home))
  }
  return send(response, 404, { error: 'Not found' })
}

export const createC4Server = () => createServer((request, response) => {
  handle(request, response).catch((error: unknown) => {
    if (error instanceof OperationError || error instanceof SyntaxError) return send(response, 422, { error: error.message })
    console.error(error)
    send(response, 500, { error: 'C4 service failed' })
  })
})

if (process.argv[1] && realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url))) {
  const port = Number(process.env.C4_SERVICE_PORT ?? 8011)
  const host = process.env.C4_SERVICE_HOST ?? '127.0.0.1'
  createC4Server().listen(port, host, () => console.log(JSON.stringify({ event: 'ready', host, port })))
}
