const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type Prediction = {
  predicted_sign: string;
  confidence: number;
  model_used: string;
  landmarks?: Array<{ x: number; y: number; z: number }>;
  bounding_box?: { x_min: number; y_min: number; x_max: number; y_max: number } | null;
  timestamp: string;
  message?: string;
};

export async function healthCheck() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}

export async function predictFrame(file: Blob): Promise<Prediction> {
  const formData = new FormData();
  formData.append("frame", file, "frame.jpg");
  const response = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
  if (!response.ok) throw new Error("Prediction request failed");
  return response.json();
}

export async function fetchHistory() {
  const response = await fetch(`${API_BASE}/history`);
  return response.json();
}

export async function fetchMetrics() {
  const response = await fetch(`${API_BASE}/metrics`);
  return response.json();
}
