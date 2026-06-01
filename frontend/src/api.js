/**
 * api.js
 * ------
 * Thin client wrapper around the Flask backend's REST API. Every function here
 * maps to exactly one endpoint defined in the project's Flask API contract.
 *
 * Design notes:
 *   - The backend base URL comes from the REACT_APP_API_URL environment
 *     variable (exposed by Create React App), falling back to localhost:5000.
 *   - The FRONTEND ALWAYS sends RAW (un-normalized) flattened landmarks.
 *     Normalization happens server-side so there is exactly one implementation.
 *   - Each helper performs a fetch, throws a descriptive Error on a non-OK
 *     response (so callers can try/catch), and returns the parsed JSON body.
 */

// Base URL of the backend. Trailing slashes are stripped so we can safely
// concatenate paths like `${BASE}/api/health`.
export const BASE = (
  process.env.REACT_APP_API_URL || "http://localhost:5000"
).replace(/\/+$/, "");

/**
 * Internal helper: perform a fetch and parse the JSON response.
 *
 * @param {string} path   API path beginning with "/" (e.g. "/api/health").
 * @param {object} [opts] Optional fetch options (method, headers, body).
 * @returns {Promise<object>} The parsed JSON body.
 * @throws {Error} If the network request fails or the response is non-OK.
 */
async function request(path, opts = {}) {
  const url = `${BASE}${path}`;

  let response;
  try {
    response = await fetch(url, opts);
  } catch (networkError) {
    // fetch only rejects on network-level failures (server down, CORS, DNS).
    throw new Error(
      `Network error contacting backend at ${url}: ${networkError.message}`
    );
  }

  // Attempt to parse the body as JSON regardless of status so that we can
  // surface backend-provided error messages where available.
  let body = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (parseError) {
      // Non-JSON body (e.g. an HTML error page). Keep the raw text for context.
      body = { raw: text };
    }
  }

  if (!response.ok) {
    const detail =
      body && (body.error || body.message)
        ? `: ${body.error || body.message}`
        : "";
    throw new Error(`Request to ${path} failed (${response.status})${detail}`);
  }

  return body;
}

/**
 * GET /api/health
 * Returns backend health information.
 *
 * @returns {Promise<{status: string, device: string,
 *                     models: {static: boolean, dynamic: boolean}}>}
 */
export async function getHealth() {
  return request("/api/health", { method: "GET" });
}

/**
 * GET /api/labels
 * Returns the class label lists for both models.
 *
 * @returns {Promise<{static: string[], dynamic: string[]}>}
 */
export async function getLabels() {
  return request("/api/labels", { method: "GET" });
}

/**
 * POST /api/predict/static
 * Predict a static ASL letter from a single hand's RAW landmarks.
 *
 * @param {number[]} landmarks Array of 63 RAW floats ([x,y,z] * 21).
 * @returns {Promise<{prediction: string, confidence: number,
 *                    top_k: {label: string, confidence: number}[]}>}
 */
export async function predictStatic(landmarks) {
  return request("/api/predict/static", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ landmarks }),
  });
}

/**
 * POST /api/predict/dynamic
 * Predict a word-level gloss from a variable-length sequence of RAW
 * two-hand landmark frames. The server pads/truncates to SEQUENCE_LENGTH.
 *
 * @param {number[][]} sequence Array of frames, each a 126-float array.
 * @returns {Promise<{prediction: string, confidence: number,
 *                    top_k: {label: string, confidence: number}[]}>}
 */
export async function predictDynamic(sequence) {
  return request("/api/predict/dynamic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
}
