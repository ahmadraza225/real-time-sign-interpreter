/**
 * SentenceBuilder.js
 * ------------------
 * Accumulates STABLE predictions into a sentence. Because the model emits a
 * prediction several times per second, we debounce: a label is only committed
 * to the sentence once it has been observed for N consecutive updates
 * (default 10). This prevents flicker/jitter from polluting the text.
 *
 * Special static labels are mapped on commit:
 *   "space"   -> append a space (" ")
 *   "del"     -> remove the last character (backspace)
 *   "nothing" -> ignored entirely (no hand / resting pose)
 *
 * The committed sentence is lifted to the parent via props (sentence /
 * setSentence) so other parts of the app (e.g. auto-speak) can react to it.
 *
 * Props:
 *   prediction:  null | { prediction: string, confidence: number, ... }
 *   sentence:    string                 // controlled value
 *   setSentence: (updater|string)=>void // controlled setter
 *   onSpeak:     (text) => void         // speak the whole sentence
 *   onCommit:    (label) => void        // notify parent a label was committed
 *   stableCount: number                 // consecutive updates required (default 10)
 *   minConfidence: number               // ignore predictions below this (default 0.6)
 */

import React, { useEffect, useRef, useState } from "react";

export default function SentenceBuilder({
  prediction,
  sentence,
  setSentence,
  onSpeak,
  onCommit,
  stableCount = 10,
  minConfidence = 0.6,
}) {
  // Tracks the label currently being "held" and how many consecutive frames
  // it has persisted for.
  const candidateRef = useRef(null);
  const candidateCountRef = useRef(0);

  // The most recently committed label, so we don't re-commit the same held
  // pose endlessly. The user must change pose (or pass through "nothing")
  // before the same label can be committed again.
  const lastCommittedRef = useRef(null);

  // Live progress toward committing the current candidate (for the meter UI).
  const [progress, setProgress] = useState(0);
  const [lastCommitted, setLastCommitted] = useState(null);

  // Keep stable refs to callbacks/values used inside the effect.
  const setSentenceRef = useRef(setSentence);
  const onCommitRef = useRef(onCommit);
  useEffect(() => {
    setSentenceRef.current = setSentence;
  }, [setSentence]);
  useEffect(() => {
    onCommitRef.current = onCommit;
  }, [onCommit]);

  // Core debounce logic: run whenever a new prediction arrives.
  useEffect(() => {
    const label =
      prediction && typeof prediction.prediction === "string"
        ? prediction.prediction
        : null;
    const confidence = prediction ? prediction.confidence || 0 : 0;

    // No confident prediction: reset the candidate streak.
    if (!label || confidence < minConfidence) {
      candidateRef.current = null;
      candidateCountRef.current = 0;
      setProgress(0);
      return;
    }

    if (label === candidateRef.current) {
      candidateCountRef.current += 1;
    } else {
      candidateRef.current = label;
      candidateCountRef.current = 1;
      // Allow the same label to be committed again only after the pose has
      // changed; reset the "last committed" guard when a NEW candidate begins.
      if (label !== lastCommittedRef.current) {
        // New, different candidate forming.
      }
    }

    setProgress(Math.min(1, candidateCountRef.current / stableCount));

    // Commit once the streak is long enough and we have not already committed
    // this exact held pose.
    if (
      candidateCountRef.current >= stableCount &&
      label !== lastCommittedRef.current
    ) {
      lastCommittedRef.current = label;
      setLastCommitted(label);

      // Apply the label to the sentence according to its meaning.
      if (label === "nothing") {
        // Ignore entirely.
      } else if (label === "space") {
        setSentenceRef.current((prev) => `${prev} `);
      } else if (label === "del") {
        setSentenceRef.current((prev) => prev.slice(0, -1));
      } else {
        // Regular letter or word gloss.
        setSentenceRef.current((prev) => prev + label);
      }

      if (onCommitRef.current) {
        onCommitRef.current(label);
      }

      // Reset the streak so the user must re-form the pose to repeat it.
      candidateCountRef.current = 0;
      setProgress(0);
    }

    // When the candidate becomes "nothing", clear the last-committed guard so
    // the next real pose (even if identical to the previous one) can commit.
    if (label === "nothing") {
      lastCommittedRef.current = null;
    }
  }, [prediction, stableCount, minConfidence]);

  // ---- Manual editing controls ----
  const handleBackspace = () => {
    setSentence((prev) => prev.slice(0, -1));
  };

  const handleClear = () => {
    setSentence("");
    lastCommittedRef.current = null;
    candidateRef.current = null;
    candidateCountRef.current = 0;
    setProgress(0);
  };

  const handleSpeak = () => {
    if (onSpeak && sentence.trim()) {
      onSpeak(sentence);
    }
  };

  const handleCopy = async () => {
    if (!sentence) {
      return;
    }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(sentence);
      }
    } catch (e) {
      // Clipboard may be unavailable (insecure context); fail silently.
    }
  };

  const progressPct = Math.round(progress * 100);

  return (
    <section className="sentence-builder" aria-label="Sentence builder">
      <h2 className="panel-title">Sentence</h2>

      {/* The accumulated sentence. role=textbox + aria-readonly because it is
          built by the model rather than typed directly. */}
      <div
        className="sentence-box"
        role="textbox"
        aria-readonly="true"
        aria-label="Built sentence"
        tabIndex={0}
      >
        {sentence ? (
          <span className="sentence-text">{sentence}</span>
        ) : (
          <span className="sentence-placeholder">
            Your interpreted sentence will appear here.
          </span>
        )}
        <span className="sentence-caret" aria-hidden="true" />
      </div>

      {/* Stability meter: how close the current held pose is to committing. */}
      <div className="stability">
        <div className="stability-row">
          <span>Hold steady to add</span>
          <span aria-hidden="true">{progressPct}%</span>
        </div>
        <div
          className="stability-bar"
          role="progressbar"
          aria-valuenow={progressPct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Stability before committing the current sign"
        >
          <div
            className="stability-fill"
            style={{ width: `${progressPct}%` }}
          />
        </div>
        {lastCommitted && (
          <p className="stability-hint" aria-live="polite">
            Added:{" "}
            <strong>
              {lastCommitted === "space"
                ? "space"
                : lastCommitted === "del"
                ? "backspace"
                : lastCommitted}
            </strong>
          </p>
        )}
      </div>

      <div className="sentence-actions">
        <button
          type="button"
          className="btn"
          onClick={handleBackspace}
          aria-label="Delete the last character"
        >
          &#9003; Backspace
        </button>
        <button
          type="button"
          className="btn"
          onClick={handleClear}
          aria-label="Clear the entire sentence"
        >
          &#128465; Clear
        </button>
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleSpeak}
          disabled={!sentence.trim()}
          aria-label="Speak the sentence aloud"
        >
          &#128264; Speak
        </button>
        <button
          type="button"
          className="btn"
          onClick={handleCopy}
          disabled={!sentence}
          aria-label="Copy the sentence to the clipboard"
        >
          &#128203; Copy
        </button>
      </div>
    </section>
  );
}
