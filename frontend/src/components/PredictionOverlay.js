/**
 * PredictionOverlay.js
 * --------------------
 * Displays the current model prediction in large, high-contrast type along
 * with a confidence bar and the top-K alternatives. The container is an
 * aria-live="polite" region so screen readers announce new predictions
 * without interrupting other speech.
 *
 * Props:
 *   prediction: null | {
 *     prediction: string,
 *     confidence: number,            // 0..1
 *     top_k: { label: string, confidence: number }[]
 *   }
 *   pending: boolean   // a request is in flight (subtle "thinking" hint)
 *   error:   string|null  // last prediction error, if any
 */

import React from "react";

// Human-friendly display for special static labels.
function displayLabel(label) {
  if (label === "space") {
    return "␣ space";
  }
  if (label === "del") {
    return "⌫ del";
  }
  if (label === "nothing") {
    return "— nothing";
  }
  return label;
}

export default function PredictionOverlay({ prediction, pending, error }) {
  const hasPrediction =
    prediction && typeof prediction.prediction === "string";
  const confidence = hasPrediction ? prediction.confidence || 0 : 0;
  const confidencePct = Math.round(confidence * 100);

  return (
    <section
      className="prediction-overlay"
      aria-label="Current prediction"
    >
      <h2 className="panel-title">Current Sign</h2>

      <div className="prediction-main" aria-live="polite">
        <div className="prediction-letter">
          {hasPrediction ? displayLabel(prediction.prediction) : "—"}
        </div>

        <div className="prediction-confidence">
          <div className="confidence-row">
            <span>Confidence</span>
            <span className="confidence-value">
              {hasPrediction ? `${confidencePct}%` : "--"}
            </span>
          </div>
          <div
            className="confidence-bar"
            role="progressbar"
            aria-valuenow={confidencePct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Prediction confidence"
          >
            <div
              className="confidence-fill"
              style={{ width: `${confidencePct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Top-K alternatives. */}
      {hasPrediction &&
        Array.isArray(prediction.top_k) &&
        prediction.top_k.length > 0 && (
          <ul className="topk-list" aria-label="Alternative predictions">
            {prediction.top_k.map((item, idx) => {
              const pct = Math.round((item.confidence || 0) * 100);
              return (
                <li key={`${item.label}-${idx}`} className="topk-item">
                  <span className="topk-label">{displayLabel(item.label)}</span>
                  <span className="topk-bar-track">
                    <span
                      className="topk-bar-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </span>
                  <span className="topk-pct">{pct}%</span>
                </li>
              );
            })}
          </ul>
        )}

      {/* Subtle status line. */}
      <p className="prediction-status" aria-live="polite">
        {error
          ? `Prediction error: ${error}`
          : pending
          ? "Recognizing..."
          : hasPrediction
          ? ""
          : "Waiting for a hand sign."}
      </p>
    </section>
  );
}
