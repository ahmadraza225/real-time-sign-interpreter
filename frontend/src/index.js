/**
 * index.js
 * --------
 * Application entry point. Create React App's webpack config looks for this
 * file, bundles it (and everything it imports), and injects the result into
 * public/index.html.
 *
 * We use the React 18 `createRoot` API and wrap the app in <React.StrictMode>
 * so that potential problems (deprecated APIs, accidental side effects) are
 * surfaced during development.
 */

import React from "react";
import ReactDOM from "react-dom/client";

import "./index.css";
import App from "./App";

// Grab the <div id="root"> defined in public/index.html.
const container = document.getElementById("root");

// Create the React 18 concurrent root and render the application tree.
const root = ReactDOM.createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
