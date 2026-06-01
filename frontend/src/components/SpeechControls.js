/**
 * SpeechControls.js
 * -----------------
 * UI for configuring text-to-speech: choose a voice, adjust rate and pitch,
 * and toggle "auto-speak" (speak each word/letter as it is committed).
 *
 * All state lives in the useSpeech hook (lifted into App); this component is
 * presentational and simply wires the controls to the provided props.
 *
 * Props:
 *   speech: object from useSpeech (voices, selectedVoice, rate, pitch, ...)
 *   autoSpeak: boolean
 *   setAutoSpeak: (boolean) => void
 */

import React from "react";

export default function SpeechControls({ speech, autoSpeak, setAutoSpeak }) {
  const {
    supported,
    voices,
    selectedVoice,
    setSelectedVoice,
    rate,
    setRate,
    pitch,
    setPitch,
    speaking,
    cancel,
  } = speech;

  if (!supported) {
    return (
      <section className="speech-controls" aria-label="Speech controls">
        <h2 className="panel-title">Speech</h2>
        <p className="notice notice-warn">
          The Web Speech API is not available in this browser, so text-to-speech
          is disabled. Try a recent version of Chrome or Edge.
        </p>
      </section>
    );
  }

  // Match the selected voice by its unique voiceURI for the <select> value.
  const selectedValue = selectedVoice ? selectedVoice.voiceURI : "";

  const handleVoiceChange = (event) => {
    const uri = event.target.value;
    const voice = voices.find((v) => v.voiceURI === uri) || null;
    setSelectedVoice(voice);
  };

  return (
    <section className="speech-controls" aria-label="Speech controls">
      <h2 className="panel-title">Speech</h2>

      <div className="control-group">
        <label htmlFor="voice-select">Voice</label>
        <select
          id="voice-select"
          className="select"
          value={selectedValue}
          onChange={handleVoiceChange}
        >
          {voices.length === 0 && <option value="">Loading voices…</option>}
          {voices.map((voice) => (
            <option key={voice.voiceURI} value={voice.voiceURI}>
              {voice.name} ({voice.lang})
              {voice.default ? " — default" : ""}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label htmlFor="rate-slider">
          Rate <span className="control-value">{rate.toFixed(2)}×</span>
        </label>
        <input
          id="rate-slider"
          className="slider"
          type="range"
          min="0.5"
          max="2"
          step="0.05"
          value={rate}
          onChange={(e) => setRate(parseFloat(e.target.value))}
          aria-valuetext={`${rate.toFixed(2)} times speed`}
        />
      </div>

      <div className="control-group">
        <label htmlFor="pitch-slider">
          Pitch <span className="control-value">{pitch.toFixed(2)}</span>
        </label>
        <input
          id="pitch-slider"
          className="slider"
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={pitch}
          onChange={(e) => setPitch(parseFloat(e.target.value))}
          aria-valuetext={`pitch ${pitch.toFixed(2)}`}
        />
      </div>

      <div className="control-group control-row">
        <label htmlFor="autospeak-toggle" className="toggle-label">
          <input
            id="autospeak-toggle"
            type="checkbox"
            checked={autoSpeak}
            onChange={(e) => setAutoSpeak(e.target.checked)}
          />
          <span>Auto-speak committed signs</span>
        </label>
      </div>

      <div className="control-group">
        <button
          type="button"
          className="btn"
          onClick={cancel}
          disabled={!speaking}
          aria-label="Stop speaking"
        >
          &#9209; Stop speaking
        </button>
        {speaking && (
          <span className="speaking-indicator" aria-live="polite">
            Speaking…
          </span>
        )}
      </div>
    </section>
  );
}
