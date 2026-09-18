import { clsx, type ClassValue } from "clsx"
import type { Message } from "./types"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}


export function updateLastMsg(
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  updater: (msg: Message) => Message,
) {
  setMessages(prev => {
    const copy = [...prev]
    const last = copy[copy.length - 1]
    if (last?.role === "assistant") copy[copy.length - 1] = updater(last)
    return copy
  })
}
