"use client";

import { useEffect, useRef, useState, KeyboardEvent } from "react";

import VoiceButton from "@/components/chat/VoiceButton";

interface Props {
  onSend: (message: string) => void;
  disabled: boolean;
}

export default function InputBar({ onSend, disabled }: Props) {
  const [text, setText] = useState("");
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
    }
  }, [text]);

  const handleSend = () => {
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText("");
    setVoiceError(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleTranscript = (transcript: string) => {
    setText((prev) => (prev ? `${prev} ${transcript}` : transcript));
    setVoiceError(null);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl space-y-2">
      <div
        className="flex items-end gap-2 rounded-2xl border border-[#e4e4ea] bg-[#fafafa] px-4 py-3 shadow-sm transition
                   focus-within:border-[#c5a6ff] focus-within:ring-2 focus-within:ring-[#c5a6ff]/25"
      >
        <VoiceButton
          onTranscript={handleTranscript}
          onError={(message) => setVoiceError(message)}
          disabled={disabled}
        />

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          disabled={disabled}
          placeholder="Ask me anything…"
          rows={1}
          className="max-h-36 min-h-[44px] flex-1 resize-none bg-transparent text-sm leading-relaxed text-[#111] outline-none placeholder:text-[#a8a8b0] disabled:opacity-50"
        />

        <button
          type="button"
          onClick={handleSend}
          disabled={!text.trim() || disabled}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#6f4ef2] text-base font-bold text-white shadow-md transition hover:bg-[#5d42d4] disabled:cursor-not-allowed disabled:opacity-35"
          aria-label="Send message"
        >
          ↑
        </button>
      </div>

      {voiceError && <p className="mt-1 px-1 text-xs text-red-500">{voiceError}</p>}

      <p className="text-center text-xs text-[#8f8f95]">
        MedCortex is for educational purposes only — not a substitute for professional medical advice.
      </p>
    </div>
  );
}
