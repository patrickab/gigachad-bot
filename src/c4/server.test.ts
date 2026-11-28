import assert from 'node:assert/strict'
import type { AddressInfo } from 'node:net'
import { after, it } from 'node:test'
import { createC4Server } from './server.ts'
import { Workspace } from './workspace.ts'

const server = createC4Server().listen(0, '127.0.0.1')
await new Promise((resolve) => server.once('listening', resolve))
after(() => server.close())

const post = async (path: string, body: unknown) => {
  const { port } = server.address() as AddressInfo
  return fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST', body: JSON.stringify(body) })
}

it('answers a repeated /model without parsing the workspace again', async () => {
  const open = Workspace.open
  let opens = 0
  Workspace.open = (sources) => (opens++, open.call(Workspace, sources))
  const body = { sources: { 'model.c4': 'specification { element system }\nmodel { a = system }\nviews { view index { include * } }\n' } }
  const first = await post('/model', body)
  assert.equal(first.status, 200)
  const second = await post('/model', body)
  assert.deepEqual(await second.json(), await first.json())
  assert.equal(opens, 1)
  assert.equal((await post('/model', { sources: { 'bad.txt': '' } })).status, 422)
  assert.equal((await post('/model', { sources: {}, snapshots: [] })).status, 422)
})
