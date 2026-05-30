import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fetchMetrics } from "../lib/api";

type MetricsPayload = {
  models: Record<string, { accuracy: number; precision: number; recall: number; f1: number }>;
  confusion_matrix: { labels: string[]; matrix: number[][] };
};

export default function AnalyticsPage() {
  const [payload, setPayload] = useState<MetricsPayload>({
    models: {
      random_forest: { accuracy: 0, precision: 0, recall: 0, f1: 0 },
      neural_network: { accuracy: 0, precision: 0, recall: 0, f1: 0 },
    },
    confusion_matrix: { labels: [], matrix: [] },
  });

  useEffect(() => {
    fetchMetrics().then(setPayload).catch(() => undefined);
  }, []);

  const chartData = useMemo(
    () => ["accuracy", "precision", "recall", "f1"].map((metric) => ({
      metric: metric.toUpperCase(),
      random_forest: payload.models.random_forest?.[metric as keyof (typeof payload.models.random_forest)] ?? 0,
      neural_network: payload.models.neural_network?.[metric as keyof (typeof payload.models.neural_network)] ?? 0,
    })),
    [payload],
  );

  return (
    <section className="space-y-6">
      <h1 className="text-3xl font-bold">Analytics</h1>
      <div className="rounded border border-slate-700 bg-slate-900 p-4">
        <h2 className="mb-3 text-xl font-semibold">Model Comparison Dashboard</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="random_forest" fill="#06b6d4" />
              <Bar dataKey="neural_network" fill="#22c55e" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="overflow-x-auto rounded border border-slate-700 bg-slate-900 p-4">
        <h2 className="mb-3 text-xl font-semibold">Confusion Matrix</h2>
        <table className="text-xs">
          <tbody>
            {payload.confusion_matrix.matrix.slice(0, 10).map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.slice(0, 10).map((value, colIndex) => (
                  <td key={colIndex} className="border border-slate-700 p-1 text-center">{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-sm text-slate-400">Showing top-left 10x10 matrix preview for readability.</p>
      </div>
    </section>
  );
}
