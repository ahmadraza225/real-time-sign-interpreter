# Sign Language Interpreter — Frontend

React (Create React App style, plain JavaScript) frontend for the Real-Time
Sign Language Interpreter university project. It captures hand landmarks in the
browser with **MediaPipe Hands**, sends the **raw** landmarks to the Flask
backend for prediction, builds a sentence from stable predictions, and reads it
aloud using the browser **Web Speech API**.

---

## Prerequisites

- **Node.js 18+** (the project targets Node 24).
- The **Flask backend** running on `http://localhost:5000` (see the project's
  `backend/` directory). The frontend works without the backend too — it will
  simply show a "Backend offline" badge and skip predictions.
- A modern browser with the Web Speech API and `getUserMedia` (recent
  **Chrome** or **Edge** recommended).

---

## Setup

From the `frontend/` directory:

```bash
npm install
```

> **Windows PowerShell users:** if `npm` errors with `running scripts is disabled
> on this system` / `npm.ps1 cannot be loaded`, that is PowerShell's execution
> policy — not a project issue. Either run
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once
> (then use `npm` normally), or just use `npm.cmd install` / `npm.cmd start`,
> which PowerShell does not block.

Copy the example environment file and adjust it if your backend runs elsewhere:

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

`.env` contents (default):

```
REACT_APP_API_URL=http://localhost:5000
```

> Only variables prefixed with `REACT_APP_` are exposed to the browser bundle by
> Create React App.

---

## Running

```bash
npm start
```

This starts the dev server and opens <http://localhost:3000>. Make sure the
backend is listening on <http://localhost:5000> (CORS is configured on the
backend to allow `http://localhost:3000`).

To create an optimized production build:

```bash
npm run build
```

---

## Camera & permissions

- On first use the browser will **prompt for camera permission**. You must
  **allow** it for live recognition to work.
- `getUserMedia` (the camera API) is only available in a **secure context**:
  either **HTTPS** or **`localhost`/`127.0.0.1`**. `http://localhost:3000`
  qualifies, so local development works out of the box. If you serve the built
  app from a remote host, it must be over **HTTPS**.
- **No webcam?** The project's desktop machine has no camera. That's fine — the
  app detects this and shows an accessible **"Camera unavailable on this
  device"** placeholder. You can still explore the entire UI, watch the backend
  health badge, and exercise the trained models via the backend test script:

  ```bash
  python -m src.inference.test_api
  ```

---

## How it works

| Concern              | Where                                   |
| -------------------- | --------------------------------------- |
| Backend API calls    | `src/api.js`                            |
| Camera + MediaPipe   | `src/hooks/useWebcam.js`                |
| Text-to-speech       | `src/hooks/useSpeech.js`                |
| Live skeleton + feed | `src/components/CameraFeed.js`          |
| Current prediction   | `src/components/PredictionOverlay.js`   |
| Sentence assembly    | `src/components/SentenceBuilder.js`     |
| Speech settings      | `src/components/SpeechControls.js`      |
| Composition / glue   | `src/App.js`                            |

### Landmark flow

1. `useWebcam` runs MediaPipe Hands on each frame and produces **raw**
   (un-normalized) flattened landmarks:
   - `flat63` — the first detected hand, `[x, y, z] * 21`.
   - `flat126` — two hands ordered **Left then Right**, zeros for a missing
     hand.
2. `App` throttles requests to **~5 per second**, always sending the *latest*
   `flat63` to `POST /api/predict/static`. The throttle reads the landmarks from
   a ref inside a `setInterval`, so it **never blocks React's render loop** and
   never queues a backlog of requests.
3. **Normalization happens server-side** — the frontend always sends raw
   landmarks so there is exactly one normalization implementation in the whole
   project.

### Sentence building

`SentenceBuilder` debounces predictions: a label is only committed once it has
been held **stable for ~10 consecutive updates** above a confidence threshold.
Special static labels are mapped on commit:

- `space` → inserts a space
- `del` → backspace (removes the last character)
- `nothing` → ignored

### Speech

`SpeechControls` lets you pick a voice and adjust rate/pitch, and toggle
**auto-speak** (speak each committed sign). The **Speak** button in the
sentence panel reads the whole sentence aloud.

---

## Accessibility

- High-contrast dark theme with large, readable type.
- `aria-live` regions for the prediction and sentence so screen readers announce
  updates.
- Visible keyboard focus outlines and labelled controls.
- Respects `prefers-reduced-motion`.

---

## Troubleshooting

- **`npm` won't run on Windows (`running scripts is disabled` / `UnauthorizedAccess`):**
  PowerShell execution policy. Run
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use
  `npm.cmd` in place of `npm`.
- **"Backend offline" badge:** ensure the Flask server is running on port 5000
  and that `REACT_APP_API_URL` matches.
- **Predictions never appear:** the static model may not be loaded — check the
  "Static model" status in the header (it reflects `GET /api/health`).
- **Camera permission denied:** clear the site's camera permission in the
  browser and reload, then allow access.
- **No voices in the dropdown:** some browsers load voices asynchronously;
  give it a moment, or reload the page.
