import { useEffect, useMemo, useRef, useState } from "react";
import { predictFrame, type Prediction } from "../lib/api";

export default function LiveInterpreterPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [running, setRunning] = useState(false);
  const [latest, setLatest] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream;
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
        }
      })
      .catch(() => setError("Webcam access denied or unavailable."));

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(async () => {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      const overlay = overlayCanvasRef.current;
      if (!video || !canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      if (overlay) {
        overlay.width = canvas.width;
        overlay.height = canvas.height;
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
      if (!blob) return;
      try {
        const prediction = await predictFrame(blob);
        setLatest(prediction);
        setHistory((prev) => [prediction, ...prev].slice(0, 25));
        if (overlay) {
          const overlayCtx = overlay.getContext("2d");
          if (overlayCtx) {
            overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
            if (prediction.bounding_box) {
              overlayCtx.strokeStyle = "#00e5ff";
              overlayCtx.lineWidth = 2;
              overlayCtx.strokeRect(
                prediction.bounding_box.x_min * overlay.width,
                prediction.bounding_box.y_min * overlay.height,
                (prediction.bounding_box.x_max - prediction.bounding_box.x_min) * overlay.width,
                (prediction.bounding_box.y_max - prediction.bounding_box.y_min) * overlay.height,
              );
            }
            prediction.landmarks?.forEach((point) => {
              overlayCtx.fillStyle = "#22c55e";
              overlayCtx.beginPath();
              overlayCtx.arc(point.x * overlay.width, point.y * overlay.height, 3, 0, Math.PI * 2);
              overlayCtx.fill();
            });
          }
        }
      } catch {
        setError("Prediction request failed.");
      }
    }, 1300);

    return () => clearInterval(id);
  }, [running]);

  const spokenText = useMemo(() => latest?.predicted_sign ?? "", [latest]);

  const speakLatest = () => {
    if (!spokenText) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(spokenText));
  };

  return (
    <section className="space-y-6">
      <h1 className="text-3xl font-bold">Live Interpreter</h1>
      {error && <p className="rounded border border-red-700 bg-red-950 p-3 text-red-200">{error}</p>}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-3 rounded-lg border border-slate-700 bg-slate-900 p-4">
          <div className="relative">
            <video ref={videoRef} autoPlay playsInline className="w-full rounded" aria-label="Live webcam feed" />
            <canvas ref={overlayCanvasRef} className="pointer-events-none absolute left-0 top-0 h-full w-full" aria-label="Landmark overlay" />
          </div>
          <canvas ref={captureCanvasRef} className="hidden" />
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className="rounded bg-cyan-500 px-3 py-2 font-semibold text-slate-900"
              onClick={() => setRunning((prev) => !prev)}
              aria-label={running ? "Stop detection" : "Start detection"}
            >
              {running ? "Stop Detection" : "Start Detection"}
            </button>
            <button
              type="button"
              className="rounded border border-slate-600 px-3 py-2"
              onClick={speakLatest}
              aria-label="Speak detected sign"
            >
              Speak Text
            </button>
          </div>
        </div>

        <div className="space-y-3 rounded-lg border border-slate-700 bg-slate-900 p-4">
          <h2 className="text-xl font-semibold">Prediction Output</h2>
          <p>
            <span className="font-semibold">Current Sign:</span> {latest?.predicted_sign ?? "—"}
          </p>
          <p>
            <span className="font-semibold">Confidence:</span> {latest ? `${(latest.confidence * 100).toFixed(1)}%` : "—"}
          </p>
          <p>
            <span className="font-semibold">Model:</span> {latest?.model_used ?? "—"}
          </p>
          <h3 className="pt-2 font-semibold">Detection History</h3>
          <ul className="max-h-56 space-y-2 overflow-auto" aria-label="Detection history">
            {history.map((item, index) => (
              <li key={`${item.timestamp}-${index}`} className="rounded border border-slate-700 p-2 text-sm">
                {new Date(item.timestamp).toLocaleTimeString()} — {item.predicted_sign} ({(item.confidence * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
