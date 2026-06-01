/**
 * Header.js
 * ---------
 * Top application banner showing the app title and a live backend health
 * badge. The badge reflects whether the Flask backend is reachable and which
 * models it currently has loaded.
 *
 * The health object is provided by the parent (App) which polls /api/health
 * on an interval. Passing it down (rather than fetching here) keeps a single
 * source of truth and avoids duplicate polling.
 *
 * Props:
 *   health: null | {
 *     status: string,
 *     device: string,
 *     models: { static: boolean, dynamic: boolean }
 *   }
 *   online: boolean  // whether the last health check succeeded
 */

import React from "react";

export default function Header({ health, online }) {
  // Derive a concise status label + colour class.
  let statusLabel;
  let statusClass;
  if (online && health) {
    statusLabel = "Backend online";
    statusClass = "badge badge-online";
  } else {
    statusLabel = "Backend offline";
    statusClass = "badge badge-offline";
  }

  const device = health && health.device ? health.device : "unknown";
  const staticLoaded = !!(health && health.models && health.models.static);
  const dynamicLoaded = !!(health && health.models && health.models.dynamic);

  return (
    <header className="app-header" role="banner">
      <div className="app-header-title">
        <span className="app-logo" aria-hidden="true">
          {/* Simple "signing hand" glyph for visual identity. */}
          &#9995;
        </span>
        <div>
          <h1>Sign Language Interpreter</h1>
          <p className="app-subtitle">
            Real-time ASL recognition &amp; speech
          </p>
        </div>
      </div>

      <div
        className="app-header-status"
        role="status"
        aria-live="polite"
        aria-label={`Backend status: ${statusLabel}`}
      >
        <span className={statusClass}>
          <span className="badge-dot" aria-hidden="true" />
          {statusLabel}
        </span>

        {online && health && (
          <span className="badge-meta">
            <span className="badge-meta-item" title="Inference device">
              Device: <strong>{device}</strong>
            </span>
            <span
              className={
                staticLoaded
                  ? "badge-meta-item model-on"
                  : "badge-meta-item model-off"
              }
              title="Static (letter) model status"
            >
              Static model: <strong>{staticLoaded ? "ready" : "not loaded"}</strong>
            </span>
            <span
              className={
                dynamicLoaded
                  ? "badge-meta-item model-on"
                  : "badge-meta-item model-off"
              }
              title="Dynamic (word) model status"
            >
              Word model: <strong>{dynamicLoaded ? "ready" : "not loaded"}</strong>
            </span>
          </span>
        )}
      </div>
    </header>
  );
}
