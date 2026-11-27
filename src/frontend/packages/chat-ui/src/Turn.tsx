import { Children, type CSSProperties, type ReactNode, type Ref } from "react"
import { Bot, Check, ChevronDown, ChevronRight, Copy, GitFork, User, Wrench } from "lucide-react"
import { useCopyFeedback } from "./useCopyFeedback.ts"

const ACTION_SIZE_PX = 22
const ACTION_GAP_PX = 4
const ACTION_END_GUTTER_PX = 12

export type TurnRole = "question" | "answer"

export interface TurnPreviewTool {
  key: string
  label: string
  /** Defaults to a wrench. */
  icon?: ReactNode
}

/** Card around one question and its answer. Pair with `turn.css`. */
export function TurnCard({ expanded, focused, ref, children }: {
  expanded: boolean
  /** Keyboard focus is on this turn. */
  focused?: boolean
  ref?: Ref<HTMLDivElement>
  children: ReactNode
}) {
  return (
    <div ref={ref} className="gc-turn" data-expanded={expanded} data-focused={focused || undefined}>
      {children}
    </div>
  )
}

/** Collapsed header: avatar (the question's unless `icon` is given), one-line label and the expand chevron. */
export function TurnHeader({ label, icon, onExpand, ref }: { label: string; icon?: ReactNode; onExpand: () => void; ref?: Ref<HTMLButtonElement> }) {
  return (
    <button type="button" ref={ref} className="gc-turn-header" aria-expanded={false} onClick={onExpand}>
      <span className="gc-turn-avatar">{icon ?? <User size={14} aria-hidden="true" />}</span>
      <span className="gc-turn-label">{label}</span>
      <ChevronRight className="gc-turn-chevron" size={16} aria-hidden="true" />
    </button>
  )
}

/** Collapsed answer: one row per tool, then the answer in a scrollable surface that never toggles the card. */
export function TurnPreview({ tools, children }: { tools: TurnPreviewTool[]; children?: ReactNode }) {
  if (tools.length === 0 && !children) return null
  return (
    <div className="gc-turn-preview">
      {tools.length > 0 && (
        <ul className="gc-turn-tools" aria-label="Tools used">
          {tools.map((tool) => (
            <li key={tool.key} className="gc-turn-tool">
              {tool.icon ?? <Wrench size={14} aria-hidden="true" />}
              <span>{tool.label}</span>
            </li>
          ))}
        </ul>
      )}
      {children && (
        <div className="gc-turn-surface">
          <div className="gc-turn-preview-scroll" role="region" aria-label="Answer preview" tabIndex={0}>{children}</div>
        </div>
      )}
    </div>
  )
}

/** Inset surface inside an expanded card: the answer, or a tool call. */
export function TurnSurface({ children }: { children: ReactNode }) {
  return <div className="gc-turn-surface">{children}</div>
}

/**
 * Avatar, content and hover actions for one side of a turn.
 *
 * - `tail="text"`: the content may end in a paragraph, whose last line then also holds the actions
 * - `onCollapse`: makes the avatar, the strip above the content and the chevron collapse the turn, the content stays selectable
 * - `icon` replaces the role's avatar, for a question that is no typed prompt
 */
export function TurnRegion({ role, tail = "text", actions, onCollapse, icon, children }: {
  role: TurnRole
  tail?: "text" | "activity"
  actions?: ReactNode
  onCollapse?: () => void
  icon?: ReactNode
  children: ReactNode
}) {
  const actionCount = Children.toArray(actions).length
  const actionSpace = actionCount * ACTION_SIZE_PX + Math.max(0, actionCount - 1) * ACTION_GAP_PX + ACTION_END_GUTTER_PX
  const Avatar = role === "question" ? User : Bot
  const avatarIcon = icon ?? <Avatar size={14} aria-hidden="true" />
  return (
    <div className="gc-turn-region" data-role={role}>
      {onCollapse && <span className="gc-turn-collapse-strip" aria-hidden="true" onClick={onCollapse} />}
      {onCollapse ? (
        <button type="button" className="gc-turn-avatar gc-turn-avatar-button" tabIndex={-1} onClick={onCollapse} aria-label="Collapse question and answer">
          {avatarIcon}
        </button>
      ) : (
        <span className="gc-turn-avatar">{avatarIcon}</span>
      )}
      <div className="gc-turn-body" data-tail={tail} style={{ "--gc-turn-actions-space": `${actionSpace}px` } as CSSProperties}>
        <div className="gc-turn-content">{children}</div>
        {actionCount > 0 && (
          <div className="gc-turn-actions" role="group" aria-label={role === "question" ? "Question actions" : "Answer actions"}>
            {actions}
          </div>
        )}
      </div>
      {onCollapse && (
        <button type="button" className="gc-turn-collapse" aria-expanded onClick={onCollapse} aria-label="Collapse question and answer">
          <ChevronDown size={16} aria-hidden="true" />
        </button>
      )}
    </div>
  )
}

/** Icon button in a region's action row. */
export function TurnAction({ label, icon, onClick, tone }: {
  label: string
  icon: ReactNode
  onClick: () => void
  tone?: "danger"
}) {
  return (
    <button
      type="button"
      className="gc-turn-action"
      data-tone={tone}
      aria-label={label}
      onClick={(event) => {
        event.stopPropagation()
        onClick()
      }}
    >
      {icon}
    </button>
  )
}

/** Copy `text` as markdown, confirmed by a check mark. */
export function CopyAction({ text, label }: { text: string; label: string }) {
  const { copied, copy } = useCopyFeedback(text)
  return <TurnAction label={label} onClick={copy} icon={copied ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />} />
}

/** Start a new session or branch from this point. */
export function ForkAction({ label, onFork }: { label: string; onFork: () => void }) {
  return <TurnAction label={label} onClick={onFork} icon={<GitFork size={14} aria-hidden="true" />} />
}
