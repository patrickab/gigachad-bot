import type { PlotFigure, SandboxToolResultRecord, ToolCallRecord, ToolSource } from "@/lib/domain/types"

/** A call's detail once its name has been read. `sources` cites what it read, `plot` carries a
 *  figure, `diagram` Mermaid, `mindmap` markdown, `workspace` a sandbox run, `learn` the learn
 *  session it started, and `generic` catches a name this build no longer knows (an older saved chat). */
type ToolPayload =
  | { family: "sources"; sources: ToolSource[]; costs: number | null }
  | { family: "plot"; figure: PlotFigure | null; brief: string | null; script: string | null }
  | { family: "diagram"; mermaid: string | null }
  | { family: "mindmap"; content: string | null }
  | { family: "workspace"; sandbox: SandboxToolResultRecord | null }
  | { family: "learn"; session: { sessionId: string; title: string; projectSlug: string | null } | null }
  | { family: "generic" }

/** The only place a tool name is read. Another citing tool joins the `sources` case. */
export function toolPayload(call: ToolCallRecord): ToolPayload {
  const detail = call.detail
  switch (call.name) {
    case "web_search":
    case "deep_research":
      return {
        family: "sources",
        sources: call.sources ?? [],
        costs: typeof detail?.costs === "number" ? detail.costs : null,
      }
    case "sandbox_plot":
      return {
        family: "plot",
        figure: detail?.figure && typeof detail.figure === "object" ? detail.figure : null,
        brief: detail?.brief ?? null,
        script: detail?.script ?? null,
      }
    case "diagram":
      return { family: "diagram", mermaid: typeof detail?.mermaid === "string" ? detail.mermaid : null }
    case "mindmap":
      return { family: "mindmap", content: typeof detail?.mindmap === "string" ? detail.mindmap : null }
    case "workspace_agent":
      return { family: "workspace", sandbox: call.sandbox ?? null }
    case "start_learn_session": {
      const sessionId = typeof detail?.session_id === "string" ? detail.session_id : null
      return {
        family: "learn",
        session: sessionId === null ? null : {
          sessionId,
          title: typeof detail?.title === "string" ? detail.title : sessionId,
          projectSlug: typeof detail?.project_slug === "string" ? detail.project_slug : null,
        },
      }
    }
    default:
      return { family: "generic" }
  }
}
