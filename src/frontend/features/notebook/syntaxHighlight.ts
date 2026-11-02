import { HighlightStyle, syntaxHighlighting, syntaxTree } from "@codemirror/language"
import { type Extension, Prec, type Range } from "@codemirror/state"
import { Decoration, type DecorationSet, EditorView, ViewPlugin, type ViewUpdate } from "@codemirror/view"
import { EditorExtensionRegistry, type IEditorExtensionFactory } from "@jupyterlab/codemirror"
import { tags as t } from "@lezer/highlight"

/** Neovim `@function.call`, `@function.method.call`, `@type`, `@boolean`, `@constant.builtin`, `@string.escape`, `@variable.builtin`, `@attribute` as CodeMirror tags, colors from `--nb-syn-*` in notebook-theme.css. Prec.high so it wins over JupyterLab's own highlighter, which the registry also applies. */
const style = HighlightStyle.define([
  { tag: [t.function(t.variableName), t.function(t.propertyName)], color: "var(--nb-syn-function)" },
  { tag: [t.className, t.definition(t.className)], color: "var(--nb-syn-type)" },
  { tag: [t.bool, t.escape], color: "var(--nb-syn-boolean)" },
  { tag: [t.null, t.self], color: "var(--nb-syn-constant)" },
])

/** Lezer's Python grammar has no tag for `self`/`cls` or decorator names, so they get a CSS class from the syntax tree (styled in notebook-theme.css). */
const SELF_NAMES = new Set(["self", "cls"])
const selfMark = Decoration.mark({ class: "gc-syn-self" })
const attributeMark = Decoration.mark({ class: "gc-syn-attribute" })

function buildMarks(view: EditorView): DecorationSet {
  const marks: Range<Decoration>[] = []
  for (const { from, to } of view.visibleRanges) {
    syntaxTree(view.state).iterate({
      from,
      to,
      enter: (node) => {
        if (node.name === "Decorator") {
          // `@name`, `@pkg.name`, `@name(args)`: the dotted name before any call arguments
          for (let child = node.node.firstChild; child && child.name !== "ArgList"; child = child.nextSibling) {
            if (child.name === "VariableName" || child.name === "PropertyName") marks.push(attributeMark.range(child.from, child.to))
            if (child.name === "MemberExpression" || child.name === "CallExpression") {
              child.node.cursor().iterate((inner) => {
                if (inner.name === "ArgList") return false
                if (inner.name === "VariableName" || inner.name === "PropertyName") marks.push(attributeMark.range(inner.from, inner.to))
                return undefined
              })
            }
          }
          return false
        }
        if (node.name === "VariableName" && SELF_NAMES.has(view.state.doc.sliceString(node.from, node.to))) marks.push(selfMark.range(node.from, node.to))
        return undefined
      },
    })
  }
  return Decoration.set(marks, true)
}

const nameMarks = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet
    constructor(view: EditorView) { this.decorations = buildMarks(view) }
    update(update: ViewUpdate) {
      if (update.docChanged || update.viewportChanged || syntaxTree(update.startState) !== syntaxTree(update.state)) this.decorations = buildMarks(update.view)
    }
  },
  { decorations: (plugin) => plugin.decorations },
)

const extension: Extension = [Prec.high(syntaxHighlighting(style)), Prec.high(nameMarks)]

/** Stable reference: the library rebuilds its editor features when the array identity changes. Name unique across the registry (a duplicate throws). */
export const NOTEBOOK_EDITOR_EXTENSIONS: IEditorExtensionFactory[] = [
  {
    name: "gigachad-syntax-highlight",
    factory: () => EditorExtensionRegistry.createImmutableExtension(extension),
  },
]
