"use client"

import { useCallback, useEffect, useRef, useState } from "react"

/** One recognition alternative, the browser API has no DOM lib typings. */
interface SpeechResultEvent {
  resultIndex: number
  results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }>
}

interface SpeechRecognitionLike {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: SpeechResultEvent) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike

/** Dictate into `text`: recognized speech is appended, interim words show live, `onResult` runs after each result is written. */
export function useSpeechInput(text: string, setText: (text: string) => void, onResult?: () => void) {
  const [isListening, setIsListening] = useState(false)
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  // Mirror of `text` so speech callbacks read the latest value without re-binding
  const textRef = useRef(text)
  useEffect(() => { textRef.current = text }, [text])

  // Stale `recognitionRef` checks ignore events from a recognizer that was already replaced or stopped
  const toggleListening = useCallback(() => {
    if (isListening) { recognitionRef.current?.stop(); setIsListening(false); return }
    const speechWindow = window as Window & { SpeechRecognition?: SpeechRecognitionConstructor; webkitSpeechRecognition?: SpeechRecognitionConstructor }
    const SpeechRecognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition
    if (!SpeechRecognition) { alert("Voice input is not supported."); return }
    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = "en-US"
    let finalText = textRef.current.trim()
    let interimText = ""
    recognition.onresult = (event) => {
      let interim = ""
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) finalText = finalText ? finalText + " " + result[0].transcript.trim() : result[0].transcript.trim()
        else interim += result[0].transcript
      }
      interimText = interim
      setText(finalText + (interimText ? " " + interimText : ""))
      onResult?.()
    }
    recognition.onend = () => {
      if (interimText && recognitionRef.current === recognition) {
        finalText = finalText ? finalText + " " + interimText.trim() : interimText.trim()
        setText(finalText)
      }
      if (recognitionRef.current === recognition) setIsListening(false)
    }
    recognition.onerror = () => { if (recognitionRef.current === recognition) setIsListening(false) }
    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }, [isListening, setText, onResult])

  return { isListening, toggleListening }
}
