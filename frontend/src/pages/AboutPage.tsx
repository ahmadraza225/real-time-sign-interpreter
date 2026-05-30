export default function AboutPage() {
  return (
    <section className="space-y-6">
      <h1 className="text-3xl font-bold">About</h1>
      <article className="rounded border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-xl font-semibold">Problem Statement</h2>
        <p className="mt-2 text-slate-300">Enable real-time communication support by translating ASL hand signs into text and speech.</p>
      </article>

      <article className="rounded border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-xl font-semibold">Technologies</h2>
        <p className="mt-2 text-slate-300">React, Tailwind CSS, FastAPI, MediaPipe, OpenCV, scikit-learn, TensorFlow/Keras, Recharts, Web Speech API.</p>
      </article>

      <article className="rounded border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-xl font-semibold">System Architecture</h2>
        <svg viewBox="0 0 800 260" className="mt-3 w-full rounded bg-slate-950 p-4" role="img" aria-label="System architecture diagram">
          <rect x="20" y="80" width="180" height="80" fill="#0e7490" />
          <text x="110" y="125" textAnchor="middle" fill="white">Frontend</text>
          <rect x="310" y="80" width="180" height="80" fill="#1d4ed8" />
          <text x="400" y="125" textAnchor="middle" fill="white">FastAPI</text>
          <rect x="600" y="80" width="180" height="80" fill="#15803d" />
          <text x="690" y="125" textAnchor="middle" fill="white">RF / NN Models</text>
          <line x1="200" y1="120" x2="310" y2="120" stroke="white" strokeWidth="4" />
          <line x1="490" y1="120" x2="600" y2="120" stroke="white" strokeWidth="4" />
        </svg>
      </article>

      <article className="rounded border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-xl font-semibold">Future Improvements</h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-slate-300">
          <li>Dynamic sequence gesture recognition with LSTM (Phase 2)</li>
          <li>User-specific calibration and adaptation</li>
          <li>Cloud deployment with monitoring and model registry</li>
        </ul>
      </article>

      <article className="rounded border border-slate-700 bg-slate-900 p-5">
        <h2 className="text-xl font-semibold">Team</h2>
        <p className="mt-2 text-slate-300">Add project member names, roles, and contact links for final delivery.</p>
      </article>
    </section>
  );
}
