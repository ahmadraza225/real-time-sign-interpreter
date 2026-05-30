import { useEffect, useMemo, useState } from "react";
import { fetchHistory, type Prediction } from "../lib/api";

function toCsv(items: Prediction[]) {
  const rows = ["timestamp,predicted_sign,confidence,model_used", ...items.map((item) => `${item.timestamp},${item.predicted_sign},${item.confidence},${item.model_used}`)];
  return rows.join("\n");
}

function download(name: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

export default function ResultsPage() {
  const [items, setItems] = useState<Prediction[]>([]);

  useEffect(() => {
    fetchHistory().then((payload) => setItems(payload.items ?? [])).catch(() => setItems([]));
  }, []);

  const summary = useMemo(() => {
    const total = items.length;
    const avg = total ? items.reduce((acc, item) => acc + item.confidence, 0) / total : 0;
    return { total, avg };
  }, [items]);

  return (
    <section className="space-y-6">
      <h1 className="text-3xl font-bold">Results</h1>
      <div className="grid gap-4 md:grid-cols-2">
        <article className="rounded border border-slate-700 bg-slate-900 p-4">
          <h2 className="font-semibold">Session Summary</h2>
          <p>Total predictions: {summary.total}</p>
          <p>Average confidence: {(summary.avg * 100).toFixed(1)}%</p>
        </article>
        <article className="rounded border border-slate-700 bg-slate-900 p-4">
          <h2 className="font-semibold">Downloads</h2>
          <div className="mt-2 flex gap-2">
            <button className="rounded border border-slate-600 px-3 py-2" onClick={() => download("results.csv", toCsv(items), "text/csv")}>Download CSV</button>
            <button className="rounded border border-slate-600 px-3 py-2" onClick={() => download("results.json", JSON.stringify(items, null, 2), "application/json")}>Download JSON</button>
          </div>
        </article>
      </div>

      <div className="overflow-x-auto rounded border border-slate-700 bg-slate-900 p-4">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="p-2">Time</th>
              <th className="p-2">Sign</th>
              <th className="p-2">Confidence</th>
              <th className="p-2">Model</th>
            </tr>
          </thead>
          <tbody>
            {items.slice(0, 50).map((item, idx) => (
              <tr key={`${item.timestamp}-${idx}`} className="border-b border-slate-800">
                <td className="p-2">{new Date(item.timestamp).toLocaleString()}</td>
                <td className="p-2">{item.predicted_sign}</td>
                <td className="p-2">{(item.confidence * 100).toFixed(1)}%</td>
                <td className="p-2">{item.model_used}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
