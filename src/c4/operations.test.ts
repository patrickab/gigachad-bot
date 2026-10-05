import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { readPins, snapshotPath, type Pins } from './layout.ts'
import { applyOperations, idFromTitle, OperationError, titleFromName, type ApplyResult, type Operation } from './operations.ts'

const SOURCE = `specification {
  element system
  element container
  element component
}

model {
  // Payments live here
  shop = system 'Shop' {
    api = container 'API'
  }
  bank = system 'Bank'
  shop.api -> bank 'charges'
}

views {
  view index {
    include *
  }
  view shopView of shop {
    include *
  }
  view custom {
    include shop, bank
  }
  view filtered {
    include * where kind is system
  }
}
`

const apply = (ops: Operation[], pins: Pins = {}, source = SOURCE) =>
  applyOperations({ 'model.c4': source }, pins, ops, 'model.c4')

describe('idFromTitle', () => {
  it('builds lowerCamelCase ASCII ids', () => {
    assert.equal(idFromTitle('Payment Service'), 'paymentService')
    assert.equal(idFromTitle('API  gateway (v2)'), 'apiGatewayV2')
    assert.equal(idFromTitle('Zürich Ops'), 'zurichOps')
    assert.equal(idFromTitle('3rd party'), 'element3rdParty')
    assert.equal(idFromTitle('!!!'), 'element')
  })

  it('titles files by their name', () => {
    assert.equal(titleFromName('payment-service'), 'Payment Service')
    assert.equal(titleFromName('backendApi'), 'Backend Api')
  })
})

describe('applyOperations', () => {
  it('nests a new element in a body-less parent and leaves the rest byte-identical', async () => {
    const result = await apply([
      { op: 'addElement', parent: 'bank', kind: 'container', title: 'Ledger' },
      { op: 'addElement', parent: 'bank', kind: 'container', title: 'ledger' },
    ])
    assert.deepEqual(result.created, ['bank.ledger', 'bank.ledger2'])
    assert.equal(result.sources['model.c4'], SOURCE.replace(
      "  bank = system 'Bank'\n",
      "  bank = system 'Bank' {\n    ledger = container 'Ledger'\n    ledger2 = container 'ledger'\n  }\n",
    ))
  })

  it('prunes connections and view entries left naming a removed element', async () => {
    const edited = SOURCE.replace("  bank = system 'Bank'\n", '')
    const result = await applyOperations({ 'model.c4': edited }, {}, [{ op: 'pruneDangling' }], 'model.c4')
    assert.deepEqual(result.model.errors, [])
    assert.doesNotMatch(result.sources['model.c4'], /bank/)
    assert.match(result.sources['model.c4'], /include shop\n/)
  })

  it('creates top-level elements and connections in the model block', async () => {
    const result = await apply([
      { op: 'addElement', parent: null, kind: 'system', title: 'Shop', description: '- sells\n- ships' },
      { op: 'addRelation', source: 'bank', target: 'shop2', label: "pays' out" },
    ])
    assert.deepEqual(result.created, ['shop2', null])
    const view = result.model.views.find((candidate) => candidate.id === 'index')!
    assert.deepEqual(view.edges.find((edge) => edge.id === 'bank->shop2')?.label, "pays' out")
    assert.equal(result.model.elements.find((element) => element.id === 'shop2')?.description, '- sells\n- ships')
  })

  it('refuses a delete that would leave dangling references and accepts it with the connection selected', async () => {
    await assert.rejects(apply([{ op: 'delete', elements: ['bank'], relations: [] }]), (error: unknown) =>
      error instanceof OperationError && /Select their connections too/.test(error.message))

    const { model } = await apply([])
    const relation = model.views.find((view) => view.id === 'shopView')!.edges[0].relations[0]
    // custom view names bank explicitly, so it has to go too: deletion is gated, not cascaded.
    const custom = SOURCE.replace('include shop, bank', 'include shop')
    const result = await apply([{ op: 'delete', elements: ['bank'], relations: [relation] }], {}, custom)
    assert.equal(result.sources['model.c4'], custom.replace("  bank = system 'Bank'\n  shop.api -> bank 'charges'\n", ''))
  })

  it('removes nested children with their deleted parent', async () => {
    const { model } = await apply([])
    const relation = model.views.find((view) => view.id === 'shopView')!.edges[0].relations[0]
    const source = SOURCE.replace('include shop, bank', 'include bank').replace(/  view shopView of shop \{\n    include \*\n  \}\n/, '')
    const result = await apply([{ op: 'delete', elements: ['shop', 'shop.api'], relations: [relation] }], {}, source)
    assert.deepEqual(result.model.elements.map((element) => element.id), ['bank'])
  })

  it('renames an element, its references, and its saved positions', async () => {
    const pinned = await apply([{ op: 'layout', view: 'shopView', nodes: { 'shop.api': { x: 1, y: 2 } }, edges: { 'shop.api->bank': { bend: 12 } } }])
    const result = await apply([{ op: 'rename', element: 'shop', id: 'store' }], readPins(pinned.snapshots))
    assert.equal(result.sources['model.c4'], SOURCE
      .replace("shop = system 'Shop'", "store = system 'Shop'")
      .replace('shop.api -> bank', 'store.api -> bank')
      .replace('of shop', 'of store')
      .replace('include shop, bank', 'include store, bank'))
    const pins = readPins(result.snapshots).shopView!
    assert.deepEqual([pins.nodes['store.api']?.x, pins.nodes['store.api']?.y], [1, 2])
    assert.deepEqual(pins.edges, { 'store.api->bank': 12 })
  })

  it('rejects a rename onto a sibling id', async () => {
    await assert.rejects(apply([{ op: 'rename', element: 'shop', id: 'bank' }]), /already exists/)
  })

  it('moves an element under a new parent with reindented text and rewritten references', async () => {
    const source = SOURCE.replace("  bank = system 'Bank'\n", "  bank = system 'Bank'\n  group = system 'Group'\n")
    const result = await apply([{ op: 'reparent', element: 'shop.api', parent: 'group' }], {}, source)
    assert.equal(result.sources['model.c4'], source
      .replace("  shop = system 'Shop' {\n    api = container 'API'\n  }\n", "  shop = system 'Shop' {\n  }\n")
      .replace("  group = system 'Group'\n", "  group = system 'Group' {\n    api = container 'API'\n  }\n")
      .replace('shop.api -> bank', 'group.api -> bank'))

    const back = await applyOperations(result.sources, {}, [{ op: 'reparent', element: 'group.api', parent: null }], 'model.c4')
    assert.ok(back.model.elements.some((element) => element.id === 'api'))
    assert.match(back.sources['model.c4'], /\n  api = container 'API'\n/)
    assert.match(back.sources['model.c4'], /\n  api -> bank 'charges'\n/)
  })

  it('moves a subtree and rewrites references into it', async () => {
    const source = SOURCE.replace("  bank = system 'Bank'\n", "  bank = system 'Bank'\n  group = system 'Group'\n")
    const result = await apply([{ op: 'reparent', element: 'shop', parent: 'group' }], {}, source)
    assert.ok(result.model.elements.some((element) => element.id === 'group.shop.api'))
    assert.match(result.sources['model.c4'], /group\.shop\.api -> bank/)
    assert.match(result.sources['model.c4'], /view shopView of group\.shop/)
    assert.match(result.sources['model.c4'], /  group = system 'Group' \{\n    shop = system 'Shop' \{\n      api = container 'API'\n    \}\n  \}/)
  })

  it('refuses to move an element into its own subtree', async () => {
    await assert.rejects(apply([{ op: 'reparent', element: 'shop', parent: 'shop.api' }]), /into itself/)
  })

  it('edits titles, descriptions, kinds, and labels in place', async () => {
    const { model } = await apply([])
    const relation = model.views.find((view) => view.id === 'shopView')!.edges[0].relations[0]
    const result = await apply([
      { op: 'setTitle', element: 'shop', title: "Bob's Shop" },
      { op: 'setDescription', element: 'shop', description: '- sells\n- ships' },
      { op: 'setKind', element: 'shop.api', kind: 'component' },
      { op: 'setLabel', relation, label: 'settles' },
    ])
    const shop = result.model.elements.find((element) => element.id === 'shop')!
    assert.equal(shop.title, "Bob's Shop")
    assert.equal(shop.description, '- sells\n- ships')
    assert.equal(result.model.elements.find((element) => element.id === 'shop.api')!.kind, 'component')
    assert.match(result.sources['model.c4'], /shop = system 'Bob\\'s Shop' \{\n    description '''\n      - sells\n      - ships\n    '''\n    api = component 'API'\n/)
    assert.match(result.sources['model.c4'], /shop\.api -> bank 'settles'/)

    const cleared = await applyOperations(result.sources, {}, [{ op: 'setDescription', element: 'shop', description: '' }], 'model.c4')
    assert.equal(cleared.model.elements.find((element) => element.id === 'shop')!.description, '')
    assert.doesNotMatch(cleared.sources['model.c4'], /description/)
  })

  it('marks views built from element includes and excludes as editable', async () => {
    const { model } = await apply([])
    assert.deepEqual(Object.fromEntries(model.views.map((view) => [view.id, [view.editable, view.scope, view.file]])), {
      index: [true, null, 'model.c4'],
      shopView: [true, 'shop', 'model.c4'],
      custom: [true, null, 'model.c4'],
      filtered: [false, null, 'model.c4'],
    })
  })

  it('applies nothing when a later operation in the batch fails', async () => {
    await assert.rejects(apply([
      { op: 'addElement', parent: null, kind: 'system', title: 'Fine' },
      { op: 'addElement', parent: null, kind: 'nope', title: 'Broken' },
    ]), /Unknown element kind/)
  })

  it('writes top-level elements into the home file and connections beside their source', async () => {
    const sources = { 'model.c4': SOURCE, 'backend/backend.c4': '// backend\n' }
    const result = await applyOperations(sources, {}, [
      { op: 'addElement', parent: null, kind: 'system', title: 'Queue' },
      { op: 'addRelation', source: 'queue', target: 'shop', label: 'feeds' },
      { op: 'addRelation', source: 'shop', target: 'queue', label: 'enqueues' },
      { op: 'addElement', parent: 'bank', kind: 'container', title: 'Ledger' },
    ], 'backend/backend.c4')
    assert.equal(result.sources['backend/backend.c4'], "// backend\nmodel {\n  queue = system 'Queue'\n  queue -> shop 'feeds'\n}\n")
    assert.match(result.sources['model.c4']!, /  shop\.api -> bank 'charges'\n  shop -> queue 'enqueues'\n\}/)
    assert.match(result.sources['model.c4']!, /bank = system 'Bank' \{\n {4}ledger = container 'Ledger'/)
    await assert.rejects(applyOperations(sources, {}, [{ op: 'addElement', parent: null, kind: 'system', title: 'X' }], null), /Open a file to draw into/)
  })
})

describe('packages, modules, and views', () => {
  const start = async () => {
    const first = await applyOperations({}, {}, [{ op: 'createPackage', title: 'Backend' }], null)
    return applyOperations(first.sources, {}, [{ op: 'createPackage', title: 'Frontend' }], null)
  }

  it('starts a package as a folder with one system and its view, and a module as an extend beside it', async () => {
    const { sources, created, model } = await start()
    assert.deepEqual(created, ['frontend/frontend.c4'])
    assert.match(sources['backend/backend.c4']!, /^specification \{/)
    assert.equal(sources['frontend/frontend.c4'], "model {\n  frontend = system 'Frontend'\n}\n\nviews {\n  view frontend of frontend {\n    include *\n  }\n}\n")
    const view = model.views.find((candidate) => candidate.id === 'frontend')!
    assert.deepEqual([view.file, view.scope, view.editable, view.manual], ['frontend/frontend.c4', 'frontend', true, false])
    // Names stay unique against elements and folders alike, and `views/` holds only views.
    const again = await applyOperations(sources, {}, [{ op: 'createPackage', title: 'Frontend' }, { op: 'createPackage', title: 'Views' }, { op: 'createView', title: 'Overview' }], null)
    assert.deepEqual(again.created.slice(0, 2), ['frontend2/frontend2.c4', 'views2/views2.c4'])

    const modules = await applyOperations(sources, {}, [
      { op: 'createModule', package: 'backend', title: 'API' },
      { op: 'createModule', package: 'backend', title: 'API' },
    ], null)
    assert.deepEqual(modules.created, ['backend/api.c4', 'backend/api2.c4'])
    assert.equal(modules.sources['backend/api.c4'], "model {\n  extend backend {\n    api = container 'API'\n  }\n}\n")
    assert.deepEqual(modules.model.views.find((candidate) => candidate.id === 'backend')!.nodes.map((node) => node.id).sort(), ['backend', 'backend.api', 'backend.api2'])
    await assert.rejects(applyOperations(modules.sources, {}, [{ op: 'createModule', package: 'backend.api', title: 'X' }], null), /top-level package/)
  })

  it('lists files by what they declare, and a module opens in its package view', async () => {
    const { sources } = await start()
    const result = await applyOperations({
      ...sources,
      'backend/api.c4': "model {\n  extend backend {\n    api = container 'API'\n  }\n}\n",
      'loose.c4': "model {\n  a = system 'A'\n  b = system 'B'\n}\n",
      'views/overview.c4': 'views {\n  view overview {\n    include *\n  }\n}\n',
    }, {}, [
      // Drawn inside the module's frame in the package view, a component lands in the module file.
      { op: 'addElement', parent: 'backend.api', kind: 'component', title: 'Auth', layout: { view: 'backend', x: 0, y: 0 } },
    ], null)
    assert.match(result.sources['backend/api.c4']!, /api = container 'API' \{\n {6}auth = component 'Auth'\n {4}\}/)
    assert.deepEqual(result.model.tree, [
      { path: 'backend/api.c4', role: 'module', element: 'backend.api', view: 'backend' },
      { path: 'backend/backend.c4', role: 'package', element: 'backend', view: 'backend' },
      { path: 'frontend/frontend.c4', role: 'package', element: 'frontend', view: 'frontend' },
      { path: 'loose.c4', role: 'file', element: null, view: null },
      { path: 'views/overview.c4', role: 'view', element: null, view: 'overview' },
    ])
  })

  it("draws into the view's system, stores a connection once, and shows it from both sides", async () => {
    const { sources } = await start()
    const result = await applyOperations(sources, {}, [
      { op: 'addElement', parent: null, kind: 'container', title: 'Web', layout: { view: 'frontend', x: 40, y: 60 } },
      { op: 'addRelation', source: 'frontend.web', target: 'backend', label: 'calls' },
    ], 'frontend/frontend.c4')
    assert.deepEqual(result.created, ['frontend.web', null])
    assert.match(result.sources['frontend/frontend.c4']!, /frontend = system 'Frontend' \{\n    web = container 'Web'\n  \}\n  frontend\.web -> backend 'calls'\n\}/)
    assert.equal(result.sources['backend/backend.c4'], sources['backend/backend.c4'])
    const edges = (id: string) => result.model.views.find((view) => view.id === id)!.edges.map((edge) => edge.id)
    assert.deepEqual(edges('frontend'), ['frontend.web->backend'])
    assert.deepEqual(edges('backend'), ['frontend->backend'])
    // Drawing at a spot pins the view: every node is saved, the drawn one where it was drawn.
    const pins = readPins(result.snapshots).frontend!
    assert.deepEqual([pins.nodes['frontend.web']?.x, pins.nodes['frontend.web']?.y], [40, 60])
    assert.deepEqual(Object.keys(result.snapshots), [snapshotPath('frontend')])
  })

  it('creates saved views, adds whole systems or single elements, and removes them again', async () => {
    const { sources } = await start()
    const seeded = await applyOperations(sources, {}, [
      { op: 'addElement', parent: 'backend', kind: 'container', title: 'API' },
      { op: 'addElement', parent: 'frontend', kind: 'container', title: 'Web' },
      { op: 'createView', title: 'Checkout flow' },
    ], null)
    assert.deepEqual(seeded.created, ['backend.api', 'frontend.web', 'views/checkoutFlow.c4'])
    assert.equal(seeded.sources['views/checkoutFlow.c4'], "views {\n  view checkoutFlow {\n    title 'Checkout flow'\n  }\n}\n")
    const nodes = (model: typeof seeded.model) => model.views.find((view) => view.id === 'checkoutFlow')!.nodes.map((node) => node.id).sort()
    assert.deepEqual(nodes(seeded.model), [])

    const added = await applyOperations(seeded.sources, {}, [
      { op: 'includeInView', view: 'checkoutFlow', elements: ['backend'], descendants: true },
      { op: 'includeInView', view: 'checkoutFlow', elements: ['frontend.web'] },
    ], null)
    assert.equal(added.sources['views/checkoutFlow.c4'], "views {\n  view checkoutFlow {\n    title 'Checkout flow'\n    include backend, backend.**\n    include frontend.web\n  }\n}\n")
    assert.deepEqual(nodes(added.model), ['backend', 'backend.api', 'frontend.web'])

    // Its own include goes; what a wider rule still draws is excluded by name; an exclusion is undone by including again.
    const removed = await applyOperations(added.sources, {}, [{ op: 'removeFromView', view: 'checkoutFlow', elements: ['frontend.web', 'backend.api'] }], null)
    assert.equal(removed.sources['views/checkoutFlow.c4'], "views {\n  view checkoutFlow {\n    title 'Checkout flow'\n    include backend, backend.**\n    exclude backend.api\n  }\n}\n")
    assert.deepEqual(nodes(removed.model), ['backend'])
    const back = await applyOperations(removed.sources, {}, [{ op: 'includeInView', view: 'checkoutFlow', elements: ['backend.api'] }], null)
    assert.equal(back.sources['views/checkoutFlow.c4'], "views {\n  view checkoutFlow {\n    title 'Checkout flow'\n    include backend, backend.**\n  }\n}\n")

    // An element drawn into a saved view is included by name, and only the drawing is reported;
    // deleting one a view names is refused.
    const drawn = await applyOperations(back.sources, {}, [{ op: 'addElement', parent: 'frontend', kind: 'container', title: 'Mobile', layout: { view: 'checkoutFlow', x: 0, y: 0 } }], null)
    assert.deepEqual(drawn.created, ['frontend.mobile'])
    assert.match(drawn.sources['views/checkoutFlow.c4']!, /include frontend\.mobile\n/)
    await assert.rejects(applyOperations(drawn.sources, {}, [{ op: 'delete', elements: ['frontend.mobile'], relations: [] }], null), /remove them from views naming them/)
    await assert.rejects(applyOperations({ 'model.c4': SOURCE }, {}, [{ op: 'includeInView', view: 'filtered', elements: ['shop'] }], null), /edit it in its source file/)
  })

  it('gives a file without a view one of its system', async () => {
    const sources = { 'model.c4': SOURCE, 'extra.c4': "model {\n  mail = system 'Mail'\n}\n" }
    const result = await applyOperations(sources, {}, [{ op: 'addFileView' }], 'extra.c4')
    assert.deepEqual(result.created, ['mail'])
    assert.equal(result.sources['extra.c4'], "model {\n  mail = system 'Mail'\n}\n\nviews {\n  view mail of mail {\n    include *\n  }\n}\n")
  })

  it('saves positions as LikeC4 snapshots: pins win, bends round-trip, gone views lose theirs', async () => {
    const pinned = await apply([{ op: 'layout', view: 'index', nodes: { bank: { x: 500, y: 40, width: 300, height: 120 } }, edges: { 'shop->bank': { bend: 30 } } }],
      { gone: { nodes: { shop: { x: 0, y: 0 } }, edges: {} } })
    assert.deepEqual(Object.keys(pinned.snapshots), [snapshotPath('index')])
    const text = pinned.snapshots[snapshotPath('index')]!
    assert.match(text, /\n  id: 'index',\n/)
    assert.match(text, /\n  _layout: 'manual',\n/)
    const pins = readPins(pinned.snapshots).index!
    assert.deepEqual(pins.nodes.bank, { x: 500, y: 40, width: 300, height: 120 })
    assert.deepEqual(pins.edges, { 'shop->bank': 30 })
    const view = pinned.model.views.find((candidate) => candidate.id === 'index')!
    assert.equal(view.manual, true)
    assert.deepEqual(view.edges.map((edge) => [edge.id, edge.bend]), [['shop->bank', 30]])
    const straight = await apply([{ op: 'layout', view: 'index', nodes: { bank: null }, edges: { 'shop->bank': null } }], readPins(pinned.snapshots))
    assert.deepEqual(readPins(straight.snapshots).index!.edges, {})
  })

  it('grows a compound around a child pinned outside it, without ever shrinking it', async () => {
    const pins: Pins = { shopView: { nodes: { shop: { x: 0, y: 0, width: 100, height: 50 }, 'shop.api': { x: 400, y: 300, width: 200, height: 100 } }, edges: {} } }
    const result = await apply([{ op: 'layout', view: 'shopView', nodes: {}, edges: {} }], pins)
    const view = result.model.views.find((candidate) => candidate.id === 'shopView')!
    const rect = (id: string) => { const node = view.nodes.find((candidate) => candidate.id === id)!; return [node.x, node.y, node.width, node.height] }
    assert.deepEqual(rect('shop'), [0, 0, 632, 432])
    assert.deepEqual(rect('shop.api'), [400, 300, 200, 100])
  })

  it('moves an unpinned child with its pinned parent, while a pinned child keeps its pin', async () => {
    const rects = (result: ApplyResult) => {
      const view = result.model.views.find((candidate) => candidate.id === 'shopView')!
      return Object.fromEntries(view.nodes.map((node) => [node.id, [node.x, node.y, node.width, node.height]]))
    }
    const auto = rects(await apply([]))
    const [shopX, shopY] = auto.shop!
    const [apiX, apiY, apiWidth, apiHeight] = auto['shop.api']!
    const moved = await apply([], { shopView: { nodes: { shop: { x: shopX! + 500, y: shopY! + 300 } }, edges: {} } })
    assert.deepEqual(rects(moved)['shop.api'], [apiX! + 500, apiY! + 300, apiWidth, apiHeight])
    // The snapshot holds the result, so reading it back changes nothing.
    assert.deepEqual(rects(await apply([], readPins(moved.snapshots))), rects(moved))
    const own = await apply([], { shopView: { nodes: { shop: { x: shopX! + 500, y: shopY! + 300 }, 'shop.api': { x: 2000, y: 1000 } }, edges: {} } })
    assert.deepEqual(rects(own)['shop.api'], [2000, 1000, apiWidth, apiHeight])
  })

  it('ignores a snapshot it cannot read, so its view is auto-laid-out', async () => {
    const pinned = await apply([{ op: 'layout', view: 'custom', nodes: { bank: { x: 0, y: 0 } }, edges: {} }])
    const snapshots = {
      ...pinned.snapshots,
      [snapshotPath('index')]: "{ nodes: [null], edges: [] }",
      [snapshotPath('shopView')]: "{ nodes: [], edges: [{ source: 'shop.api', target: 'bank', controlPoints: [{ x: 'left', y: 0 }] }] }",
    }
    const pins = readPins(snapshots)
    assert.deepEqual(Object.keys(pins), ['custom'])
    const result = await apply([], pins)
    assert.deepEqual(result.model.views.map((view) => [view.id, view.manual]), [['index', false], ['shopView', false], ['custom', true], ['filtered', false]])
  })

  it('places elements added to an arranged view beside it, not on top of it', async () => {
    const result = await apply([{ op: 'layout', view: 'index', nodes: {}, edges: {} }], { index: { nodes: { bank: { x: 0, y: 0, width: 320, height: 180 } }, edges: {} } })
    const shop = result.model.views.find((candidate) => candidate.id === 'index')!.nodes.find((node) => node.id === 'shop')!
    assert.deepEqual([shop.x, shop.y], [440, 0])
  })
})
