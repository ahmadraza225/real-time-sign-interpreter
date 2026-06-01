/**
 * useWebcam.js
 * ------------
 * Custom React hook that wires up MediaPipe Hands + the MediaPipe Camera
 * utility to a <video> element, draws the detected hand skeleton onto a
 * <canvas> overlay (mirrored, like a selfie view), and reports the RAW
 * (un-normalized) flattened landmarks back to the caller on every frame.
 *
 * The hook is intentionally "dumb" about prediction: it only extracts and
 * forwards landmarks. Normalization and model inference happen server-side.
 *
 * Returned API:
 *   {
 *     ready,   // boolean: true once the camera + detector are running
 *     error,   // string | null: human-readable error (e.g. no webcam)
 *     start,   // () => void: (re)start the camera
 *     stop     // () => void: stop the camera and release the device
 *   }
 *
 * Landmark conventions (must match the Flask API contract):
 *   - flat63:  first detected hand, RAW [x, y, z] * 21  (63 floats).
 *   - flat126: two hands ordered Left then Right, zeros(63) for a missing
 *              hand, RAW  (126 floats).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import {
  drawConnectors,
  drawLandmarks,
} from "@mediapipe/drawing_utils";

// MediaPipe ships its WASM/model assets separately from the npm package.
// We load them from the jsDelivr CDN so the app works without bundling the
// large binary assets ourselves. The version is pinned to the installed
// @mediapipe/hands package version for reproducibility.
const HANDS_VERSION = "0.4.1675469240";
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VERSION}`;

/**
 * @param {object}   params
 * @param {React.RefObject<HTMLVideoElement>}  params.videoRef  Hidden source video.
 * @param {React.RefObject<HTMLCanvasElement>} params.canvasRef Visible overlay canvas.
 * @param {(payload: {
 *            flat63: number[],
 *            flat126: number[],
 *            rawLandmarks: Array<Array<{x:number,y:number,z:number}>>,
 *            handsCount: number
 *          }) => void} params.onLandmarks  Called once per processed frame.
 */
export default function useWebcam({ videoRef, canvasRef, onLandmarks }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  // Persisted instances that must survive re-renders without re-creating.
  const handsRef = useRef(null);
  const cameraRef = useRef(null);

  // Keep the latest onLandmarks callback in a ref so the MediaPipe onResults
  // closure always calls the current version without re-initializing Hands.
  const onLandmarksRef = useRef(onLandmarks);
  useEffect(() => {
    onLandmarksRef.current = onLandmarks;
  }, [onLandmarks]);

  /**
   * Convert MediaPipe's per-hand landmark list into a flat [x,y,z]*21 array.
   * @param {Array<{x:number,y:number,z:number}>} landmarks
   * @returns {number[]} length-63 array of RAW floats
   */
  const flattenHand = useCallback((landmarks) => {
    const flat = new Array(63);
    for (let i = 0; i < 21; i += 1) {
      const lm = landmarks[i];
      flat[i * 3] = lm.x;
      flat[i * 3 + 1] = lm.y;
      flat[i * 3 + 2] = lm.z;
    }
    return flat;
  }, []);

  /**
   * MediaPipe results callback. Draws the overlay and forwards landmarks.
   * @param {object} results MediaPipe Hands results object.
   */
  const handleResults = useCallback(
    (results) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      // Match the canvas backing store to the incoming frame size so the
      // skeleton lines up with the video and stays crisp.
      const frame = results.image;
      if (frame && frame.width && frame.height) {
        if (canvas.width !== frame.width) {
          canvas.width = frame.width;
        }
        if (canvas.height !== frame.height) {
          canvas.height = frame.height;
        }
      }

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Mirror horizontally so the overlay matches a selfie-style video feed.
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);

      // Draw the camera frame underneath the skeleton.
      if (frame) {
        ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
      }

      const multiLandmarks = results.multiHandLandmarks || [];
      const handedness = results.multiHandedness || [];

      // Draw connectors + landmark dots for each detected hand.
      for (let i = 0; i < multiLandmarks.length; i += 1) {
        const landmarks = multiLandmarks[i];
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
          color: "#36f1cd",
          lineWidth: 4,
        });
        drawLandmarks(ctx, landmarks, {
          color: "#ff5d8f",
          lineWidth: 1,
          radius: 4,
        });
      }
      ctx.restore();

      // ---- Build the RAW landmark payloads for the backend ----

      // flat63: the FIRST detected hand (or zeros if none).
      let flat63 = new Array(63).fill(0);
      if (multiLandmarks.length > 0) {
        flat63 = flattenHand(multiLandmarks[0]);
      }

      // flat126: Left hand block first, Right hand block second. MediaPipe
      // reports handedness from the camera's perspective; because we treat the
      // feed as mirrored (selfie), MediaPipe's "Left"/"Right" labels already
      // correspond to the user's hands as drawn. We order strictly by the
      // reported label so the ordering is deterministic and matches the
      // server-side extract_two_hands_from_frame() contract.
      const leftBlock = new Array(63).fill(0);
      const rightBlock = new Array(63).fill(0);
      for (let i = 0; i < multiLandmarks.length; i += 1) {
        const label =
          handedness[i] && handedness[i].label ? handedness[i].label : "Right";
        const flat = flattenHand(multiLandmarks[i]);
        if (label === "Left") {
          for (let j = 0; j < 63; j += 1) {
            leftBlock[j] = flat[j];
          }
        } else {
          for (let j = 0; j < 63; j += 1) {
            rightBlock[j] = flat[j];
          }
        }
      }
      const flat126 = leftBlock.concat(rightBlock);

      // Forward to the consumer (App / CameraFeed).
      if (onLandmarksRef.current) {
        onLandmarksRef.current({
          flat63,
          flat126,
          rawLandmarks: multiLandmarks,
          handsCount: multiLandmarks.length,
        });
      }
    },
    [canvasRef, flattenHand]
  );

  /**
   * Lazily create the MediaPipe Hands detector (once) and register the
   * results callback.
   * @returns {Hands}
   */
  const getHands = useCallback(() => {
    if (handsRef.current) {
      return handsRef.current;
    }
    const hands = new Hands({
      locateFile: (file) => `${CDN_BASE}/${file}`,
    });
    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });
    hands.onResults(handleResults);
    handsRef.current = hands;
    return hands;
  }, [handleResults]);

  /**
   * Start (or restart) the camera + detection pipeline.
   */
  const start = useCallback(async () => {
    setError(null);

    const video = videoRef.current;
    if (!video) {
      setError("Internal error: video element is not mounted.");
      return;
    }

    // getUserMedia is only available in secure contexts (https or localhost).
    if (
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== "function"
    ) {
      setError(
        "Camera API unavailable. The app must be served over HTTPS or from localhost."
      );
      return;
    }

    // getHands() builds the MediaPipe Hands detector and stores it in
    // handsRef.current (used by the frame loop below); no local handle needed.
    try {
      getHands();
    } catch (initError) {
      setError(`Failed to initialize the hand detector: ${initError.message}`);
      return;
    }

    // Pre-flight: explicitly request the camera so we can give a precise,
    // accessible error message (e.g. permission denied / no device) instead
    // of a silent failure inside the Camera utility.
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      // We immediately stop these tracks; the Camera utility opens its own
      // stream below. This pre-flight is purely for clean error reporting.
      stream.getTracks().forEach((track) => track.stop());
    } catch (camError) {
      if (
        camError &&
        (camError.name === "NotAllowedError" ||
          camError.name === "SecurityError")
      ) {
        setError(
          "Camera permission was denied. Please allow camera access and try again."
        );
      } else if (
        camError &&
        (camError.name === "NotFoundError" ||
          camError.name === "DevicesNotFoundError" ||
          camError.name === "OverconstrainedError")
      ) {
        setError("No camera was found on this device.");
      } else {
        setError(
          `Camera unavailable: ${camError ? camError.message : "unknown error"}`
        );
      }
      setReady(false);
      return;
    }

    // Hook up the MediaPipe Camera utility, which pumps frames from the video
    // element into the Hands detector on each animation frame.
    try {
      const camera = new Camera(video, {
        onFrame: async () => {
          // Guard against frames arriving after teardown.
          if (handsRef.current && videoRef.current) {
            await handsRef.current.send({ image: videoRef.current });
          }
        },
        width: 640,
        height: 480,
      });
      cameraRef.current = camera;
      await camera.start();
      setReady(true);
      setError(null);
    } catch (startError) {
      setReady(false);
      setError(
        `Failed to start the camera stream: ${
          startError ? startError.message : "unknown error"
        }`
      );
    }
  }, [getHands, videoRef]);

  /**
   * Stop the camera and release the device. Safe to call multiple times.
   */
  const stop = useCallback(() => {
    if (cameraRef.current) {
      try {
        cameraRef.current.stop();
      } catch (e) {
        // Ignore stop errors; the device may already be released.
      }
      cameraRef.current = null;
    }

    // Defensively stop any tracks still attached to the video element.
    const video = videoRef.current;
    if (video && video.srcObject) {
      const stream = video.srcObject;
      if (stream && typeof stream.getTracks === "function") {
        stream.getTracks().forEach((track) => track.stop());
      }
      video.srcObject = null;
    }

    setReady(false);
  }, [videoRef]);

  // Clean up on unmount: stop the camera and close the detector.
  useEffect(() => {
    return () => {
      stop();
      if (handsRef.current) {
        try {
          handsRef.current.close();
        } catch (e) {
          // Ignore close errors during teardown.
        }
        handsRef.current = null;
      }
    };
    // We intentionally only run cleanup on unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { ready, error, start, stop };
}
