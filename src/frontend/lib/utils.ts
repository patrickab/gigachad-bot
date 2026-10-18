import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** True for key events typed into a Neovim terminal (`data-neovim-terminal`), which owns every key while focused. */
export function isNeovimKeyEvent(event: Event): boolean {
  return event.target instanceof Element && event.target.closest("[data-neovim-terminal]") !== null
}
