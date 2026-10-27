import type { Message, Usage } from "./types"

export interface ChatHistoriesResponse {
  files: string[]
}

export interface BranchMeta {
  chat_id: string | null
  parent_id: string | null
  branch_message_idx: number | null
  children: BranchChild[]
  qa_count: number
}

export interface BranchChild {
  chat_id: string
  branch_message_idx: number
}

/** A saved chat as `GET /chat-histories/<file>` returns it. */
export interface ChatHistory {
  messages: Message[]
  filename: string
  chat_id: string | null
  title: string | null
  usage: Usage | null
  parent_id: string | null
  branch_message_idx: number | null
  children: BranchChild[]
}
