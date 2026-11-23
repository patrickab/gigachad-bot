"use client"

import { Component, type ErrorInfo, type ReactNode } from "react"

interface ErrorBoundaryProps {
  children: ReactNode
  /** Label of the reset button. */
  resetLabel?: string
  /** Called on reset, before the children render again. */
  onReset?: () => void
  /** Fill the viewport instead of the parent (the app root). */
  fullscreen?: boolean
}

interface ErrorBoundaryState { error: Error | null }

/** Catches a render error in its subtree and shows a quiet fallback: the message and a button that remounts the children. */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return { error: error instanceof Error ? error : new Error(String(error)) }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught an error", error, info.componentStack)
  }

  private reset = () => {
    this.props.onReset?.()
    this.setState({ error: null })
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div role="alert" className={`flex flex-col items-center justify-center gap-3 p-6 text-center ${this.props.fullscreen ? "h-dvh" : "h-full"}`}>
        <p className="max-w-md text-sm text-muted break-words">{error.message || "Something went wrong"}</p>
        <button type="button" onClick={this.reset} className="rounded-md px-3 py-1 text-xs text-ink hover:bg-surface focus-visible:outline focus-visible:outline-2">
          {this.props.resetLabel ?? "Reload tab"}
        </button>
      </div>
    )
  }
}
