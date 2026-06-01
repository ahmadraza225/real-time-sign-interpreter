/**
 * CameraFeed.js
 * -------------
 * Renders the webcam capture surface:
 *   - a HIDDEN <video> element that is the raw camera source, and
 *   - a VISIBLE <canvas> overlay onto which useWebcam draws the mirrored
 *     camera frame plus the detected hand skeleton.
 *
 * When no webcam is available (e.g. the project's desktop machine has none) or
 * the user denies permission, this component shows an accessible
 * "Camera unavailable" placeholder card instead of a broken/black video. This
 * is important so the UI is still fully usable for demos on the webcam-less
 * desktop.
 *
 * Detected landmarks are bubbled up to the parent via the onLandmarks prop.
 *
 * Props:
 *   onLandmarks: ({flat63, flat126, rawLandmarks, handsCount}) => void
 *   active:      boolean  // whether the camera should be running
 *   onStatusChange?: ({ready, error}) => void  // optional status notifier
 */

import React, { useEffect, useRef } from "react";

import useWebcam from "../hooks/useWebcam";

export default function CameraFeed({ onLandmarks, active, onStatusChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const { ready, error, start, stop } = useWebcam({
    videoRef,
    canvasRef,
    onLandmarks,
  });

  // Start/stop the camera in response to the `active` flag.
  useEffect(() => {
    if (active) {
      start();
    } else {
      stop();
    }
    // start/stop are stable (useCallback), but listing them keeps lint happy.
  }, [active, start, stop]);

  // Notify the parent of status transitions so it can pause/resume sending
  // predictions when the camera is not actually producing frames.
  useEffect(() => {
    if (onStatusChange) {
      onStatusChange({ ready, error });
    }
  }, [ready, error, onStatusChange]);

  const showPlaceholder = !!error || (!ready && !active);

  return (
    <section className="camera-feed" aria-label="Camera feed">
      {/* Hidden source video. playsInline + muted are required for autoplay
          on most browsers; it is visually hidden but kept in the DOM. */}
      <video
        ref={videoRef}
        className="camera-source"
        autoPlay
        playsInline
        muted
        aria-hidden="true"
      />

      {/* Visible mirrored overlay. We keep it mounted even when showing the
          placeholder so the canvas ref stays valid for useWebcam. */}
      <div className="camera-stage">
        <canvas
          ref={canvasRef}
          className="camera-canvas"
          width={640}
          height={480}
          role="img"
          aria-label={
            ready
              ? "Live camera with detected hand skeleton overlay"
              : "Camera preview"
          }
          style={{ display: showPlaceholder ? "none" : "block" }}
        />

        {showPlaceholder && (
          <div className="camera-placeholder" role="status" aria-live="polite">
            <span className="camera-placeholder-icon" aria-hidden="true">
              &#128247;
            </span>
            <h3>Camera unavailable on this device</h3>
            <p>
              {error
                ? error
                : "No camera is currently active. On a machine without a webcam you can still explore the full interface and exercise the backend with the provided test script (src/inference/test_api.py)."}
            </p>
          </div>
        )}
      </div>

      {/* Live, polite status line for assistive technology. */}
      <p className="camera-status" aria-live="polite">
        {ready
          ? "Camera active. Show a hand sign to the camera."
          : active
          ? "Starting camera..."
          : "Camera stopped."}
      </p>
    </section>
  );
}
