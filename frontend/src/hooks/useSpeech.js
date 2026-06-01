/**
 * useSpeech.js
 * ------------
 * Custom React hook wrapping the browser's Web Speech API
 * (window.speechSynthesis) for text-to-speech.
 *
 * Handles the well-known quirks of the API:
 *   - Voices load asynchronously; we listen for the `voiceschanged` event and
 *     also poll once on mount because some browsers fire it inconsistently.
 *   - speechSynthesis can get "stuck"; speak() cancels any in-progress
 *     utterance before queueing a new one.
 *
 * Returned state + setters:
 *   {
 *     supported,        // boolean: is speechSynthesis available?
 *     voices,           // SpeechSynthesisVoice[]
 *     selectedVoice,    // SpeechSynthesisVoice | null
 *     setSelectedVoice, // (voice) => void
 *     rate,             // number (0.5 - 2)
 *     setRate,          // (n) => void
 *     pitch,            // number (0 - 2)
 *     setPitch,         // (n) => void
 *     speaking,         // boolean: is an utterance currently playing?
 *     speak,            // (text) => void
 *     cancel            // () => void
 *   }
 */

import { useCallback, useEffect, useRef, useState } from "react";

// Feature-detect once at module load.
const SUPPORTED =
  typeof window !== "undefined" &&
  "speechSynthesis" in window &&
  typeof window.SpeechSynthesisUtterance !== "undefined";

export default function useSpeech() {
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [rate, setRate] = useState(1);
  const [pitch, setPitch] = useState(1);
  const [speaking, setSpeaking] = useState(false);

  // Track whether the user has explicitly chosen a voice so that voice-list
  // refreshes don't override their choice.
  const userPickedVoiceRef = useRef(false);

  // ---- Load voices (async) ----
  useEffect(() => {
    if (!SUPPORTED) {
      return undefined;
    }

    const synth = window.speechSynthesis;

    const loadVoices = () => {
      const list = synth.getVoices();
      if (list && list.length > 0) {
        setVoices(list);
        // Auto-select a sensible default the first time voices arrive:
        // prefer an English voice, otherwise the first available.
        if (!userPickedVoiceRef.current) {
          const englishDefault =
            list.find((v) => v.default && /^en/i.test(v.lang)) ||
            list.find((v) => /^en/i.test(v.lang)) ||
            list.find((v) => v.default) ||
            list[0];
          setSelectedVoice(englishDefault || null);
        }
      }
    };

    // Initial attempt (some browsers have voices ready synchronously).
    loadVoices();

    // Subscribe to the async event for browsers that populate later.
    synth.addEventListener("voiceschanged", loadVoices);

    return () => {
      synth.removeEventListener("voiceschanged", loadVoices);
    };
  }, []);

  // Wrap setSelectedVoice so we can record that the user made an explicit pick.
  const chooseVoice = useCallback((voice) => {
    userPickedVoiceRef.current = true;
    setSelectedVoice(voice);
  }, []);

  /**
   * Cancel any in-progress or queued speech.
   */
  const cancel = useCallback(() => {
    if (!SUPPORTED) {
      return;
    }
    try {
      window.speechSynthesis.cancel();
    } catch (e) {
      // Ignore; nothing to cancel.
    }
    setSpeaking(false);
  }, []);

  /**
   * Speak the given text. Cancels any current utterance first so rapid calls
   * (e.g. committing words quickly) do not pile up.
   * @param {string} text
   */
  const speak = useCallback(
    (text) => {
      if (!SUPPORTED || !text || !String(text).trim()) {
        return;
      }
      const synth = window.speechSynthesis;

      // Clear any stuck/queued speech to keep behaviour predictable.
      try {
        synth.cancel();
      } catch (e) {
        // Ignore.
      }

      const utterance = new window.SpeechSynthesisUtterance(String(text));
      if (selectedVoice) {
        utterance.voice = selectedVoice;
        utterance.lang = selectedVoice.lang;
      }
      utterance.rate = rate;
      utterance.pitch = pitch;

      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);

      try {
        synth.speak(utterance);
      } catch (e) {
        setSpeaking(false);
      }
    },
    [selectedVoice, rate, pitch]
  );

  // Stop any speech if the component using the hook unmounts.
  useEffect(() => {
    return () => {
      if (SUPPORTED) {
        try {
          window.speechSynthesis.cancel();
        } catch (e) {
          // Ignore.
        }
      }
    };
  }, []);

  return {
    supported: SUPPORTED,
    voices,
    selectedVoice,
    setSelectedVoice: chooseVoice,
    rate,
    setRate,
    pitch,
    setPitch,
    speaking,
    speak,
    cancel,
  };
}
