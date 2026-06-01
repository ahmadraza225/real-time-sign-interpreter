/**
 * App.js
 * ------
 * Top-level composition of the Sign Language Interpreter frontend.
 *
 * Responsibilities:
 *   - Lay out the UI (Header + a responsive two-column grid).
 *   - Lift shared state: the built sentence and the speech configuration.
 *   - Receive RAW landmarks from <CameraFeed> and THROTTLE static prediction
 *     requests to ~5 per second by always sending the latest flat63 array.
 *   - Feed the prediction result into <PredictionOverlay> and
 *     <SentenceBuilder>.
 *   - Auto-speak a committed sign/word when the auto-speak toggle is on.
 *   - Poll /api/health every 5 seconds to drive the live status badge and to
 *     gracefully handle the backend being down (without crashing).
 *
 * IMPORTANT: The throttle must not block React's render loop. We keep the
 * "latest landmarks" in a ref and run a setInterval that fires the request;
 * the interval reads the ref, so renders are never blocked and we never queue
 * a backlog of requests.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";

import Header from "./components/Header";
import CameraFeed from "./components/CameraFeed";
import PredictionOverlay from "./components/PredictionOverlay";
import SentenceBuilder from "./components/SentenceBuilder";
import SpeechControls from "./components/SpeechControls";

import useSpeech from "./hooks/useSpeech";
import { getHealth, predictStatic } from "./api";

import "./App.css";

// Prediction throttle: send at most ~5 static requests per second.
const PREDICT_INTERVAL_MS = 200;
// Backend health poll cadence.
const HEALTH_INTERVAL_MS = 5000;

export default function App() {
  // ---- Shared / lifted state ----
  const [sentence, setSentence] = useState("");
  const [autoSpeak, setAutoSpeak] = useState(false);

  const [health, setHealth] = useState(null);
  const [online, setOnline] = useState(false);

  const [prediction, setPrediction] = useState(null);
  const [predictPending, setPredictPending] = useState(false);
  const [predictError, setPredictError] = useState(null);

  const [cameraActive, setCameraActive] = useState(true);
  const [cameraReady, setCameraReady] = useState(false);

  const speech = useSpeech();

  // Latest RAW one-hand landmarks (63 floats) and the number of hands seen.
  const latestFlat63Ref = useRef(null);
  const handsCountRef = useRef(0);

  // Guards so we never overlap requests or update state after unmount.
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // ---- Receive landmarks from the camera (cheap; just stash in a ref) ----
  const handleLandmarks = useCallback(({ flat63, handsCount }) => {
    latestFlat63Ref.current = flat63;
    handsCountRef.current = handsCount;
  }, []);

  // Track camera readiness so we don't fire predictions when it's not running.
  const handleCameraStatus = useCallback(({ ready }) => {
    setCameraReady(ready);
  }, []);

  // ---- Throttled static prediction loop ----
  useEffect(() => {
    const interval = setInterval(async () => {
      // Skip if: a request is already in flight, the camera isn't producing
      // frames, no hand is present, or the static model isn't loaded.
      if (inFlightRef.current) {
        return;
      }
      if (!cameraReady) {
        return;
      }
      if (handsCountRef.current < 1) {
        return;
      }
      const flat63 = latestFlat63Ref.current;
      if (!flat63 || flat63.length !== 63) {
        return;
      }
      if (!(health && health.models && health.models.static)) {
        return;
      }

      inFlightRef.current = true;
      if (mountedRef.current) {
        setPredictPending(true);
      }
      try {
        const result = await predictStatic(flat63);
        if (mountedRef.current) {
          setPrediction(result);
          setPredictError(null);
        }
      } catch (err) {
        if (mountedRef.current) {
          setPredictError(err.message);
        }
      } finally {
        inFlightRef.current = false;
        if (mountedRef.current) {
          setPredictPending(false);
        }
      }
    }, PREDICT_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [cameraReady, health]);

  // ---- Health polling ----
  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const h = await getHealth();
        if (!cancelled && mountedRef.current) {
          setHealth(h);
          setOnline(true);
        }
      } catch (err) {
        if (!cancelled && mountedRef.current) {
          setOnline(false);
          // Keep last-known health for context but mark offline via `online`.
        }
      }
    };

    poll(); // immediate first check
    const interval = setInterval(poll, HEALTH_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // ---- Auto-speak: speak a label as soon as it is committed ----
  const handleCommit = useCallback(
    (label) => {
      if (!autoSpeak) {
        return;
      }
      // Speak something meaningful for special labels.
      if (label === "nothing") {
        return;
      }
      const spoken =
        label === "space" ? "space" : label === "del" ? "delete" : label;
      speech.speak(spoken);
    },
    [autoSpeak, speech]
  );

  // Speak the entire sentence on demand (from SentenceBuilder's Speak button).
  const handleSpeakSentence = useCallback(
    (text) => {
      speech.speak(text);
    },
    [speech]
  );

  return (
    <div className="app">
      <Header health={health} online={online} />

      {/* Demo banner: important context for the webcam-less desktop. */}
      <div className="demo-banner" role="note">
        <strong>Demo mode:</strong> This interface needs a webcam for live
        recognition. On a machine without a camera you can still explore the
        entire UI; to exercise the trained models, run the backend test script{" "}
        <code>python -m src.inference.test_api</code>. The backend status badge
        above shows whether <code>:5000</code> is reachable.
      </div>

      <main className="app-main">
        {/* Left column: camera + live prediction. */}
        <div className="column column-left">
          <div className="panel">
            <div className="panel-header-row">
              <h2 className="panel-title">Camera</h2>
              <button
                type="button"
                className="btn btn-small"
                onClick={() => setCameraActive((a) => !a)}
                aria-pressed={cameraActive}
              >
                {cameraActive ? "Stop camera" : "Start camera"}
              </button>
            </div>
            <CameraFeed
              active={cameraActive}
              onLandmarks={handleLandmarks}
              onStatusChange={handleCameraStatus}
            />
          </div>

          <div className="panel">
            <PredictionOverlay
              prediction={prediction}
              pending={predictPending}
              error={predictError}
            />
          </div>
        </div>

        {/* Right column: sentence builder + speech controls. */}
        <div className="column column-right">
          <div className="panel">
            <SentenceBuilder
              prediction={prediction}
              sentence={sentence}
              setSentence={setSentence}
              onSpeak={handleSpeakSentence}
              onCommit={handleCommit}
              stableCount={10}
              minConfidence={0.6}
            />
          </div>

          <div className="panel">
            <SpeechControls
              speech={speech}
              autoSpeak={autoSpeak}
              setAutoSpeak={setAutoSpeak}
            />
          </div>
        </div>
      </main>

      <footer className="app-footer" role="contentinfo">
        <p>
          Real-Time Sign Language Interpreter — university AI project. Built with
          React, MediaPipe Hands, a PyTorch backend, and the Web Speech API.
        </p>
      </footer>
    </div>
  );
}
