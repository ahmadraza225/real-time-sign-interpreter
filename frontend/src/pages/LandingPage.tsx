import { Link } from "react-router-dom";

const features = [
  "Real-time webcam gesture interpretation",
  "Accessible UI with keyboard-first controls",
  "Speech synthesis for detected text",
  "Analytics dashboard comparing RF vs NN",
];

export default function LandingPage() {
  return (
    <section className="space-y-10">
      <div className="rounded-xl border border-slate-700 bg-gradient-to-br from-slate-900 to-slate-800 p-8">
        <p className="text-cyan-300">Breaking the communication barrier, one gesture at a time.</p>
        <h1 className="mt-3 text-4xl font-bold">Real-Time Sign Language Interpreter</h1>
        <p className="mt-4 max-w-3xl text-slate-300">
          AI-powered translation of sign gestures into readable and speakable text for inclusive communication.
        </p>
        <Link to="/interpreter" className="mt-6 inline-block rounded bg-cyan-500 px-4 py-2 font-semibold text-slate-900">
          Start Live Interpreter
        </Link>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {features.map((feature) => (
          <article key={feature} className="rounded-lg border border-slate-700 bg-slate-900 p-5">
            <h2 className="font-semibold">Feature</h2>
            <p className="mt-2 text-slate-300">{feature}</p>
          </article>
        ))}
      </div>

      <section aria-labelledby="team-heading" className="rounded-lg border border-slate-700 bg-slate-900 p-6">
        <h2 id="team-heading" className="text-2xl font-semibold">
          Team
        </h2>
        <p className="mt-2 text-slate-300">Undergraduate AI project team — add names/roles in README and About page.</p>
      </section>
    </section>
  );
}
