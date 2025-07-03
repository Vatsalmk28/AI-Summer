import React, { useState } from 'react';
import axios from 'axios';
import './css/speech.css';

function SpeechRecognition() {
  const [audioFile, setAudioFile] = useState(null);
  const [model, setModel] = useState("whisper");
  const [transcription, setTranscription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setAudioFile(e.target.files[0]);
    setTranscription("");
    setError("");
  };  

  const handleModelSelect = (selectedModel) => {
    setModel(selectedModel);
  };

  const handleUpload = async () => {
    if (!audioFile) {
      setError("Please upload an audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("model", model);

    try {
      setLoading(true);
      const response = await axios.post("http://localhost:5000/speech", formData);
      setTranscription(response.data.transcription);
    } catch (err) {
      setError(err.response?.data?.error || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="speech-container">
      <div className="wave-bg"></div>
      <div className="speech-card">
        <h2 className="speech-title">🎤 Speech Recognition</h2>

        <input type="file" accept=".mp3,.wav,.m4a" onChange={handleFileChange} className="file-input" />

        <div className="model-selector-cards">
          <div
            className={`model-card ${model === "whisper" ? "selected wobble" : ""}`}
            onClick={() => handleModelSelect("whisper")}
          >
            <h3>Whisper</h3>
            <p>High-quality transcription using Whisper model.</p>
          </div>

          <div
            className={`model-card ${model === "wav2vec2" ? "selected wobble" : ""}`}
            onClick={() => handleModelSelect("wav2vec2")}
          >
            <h3>Wav2Vec2</h3>
            <p>Fast transcription with Wav2Vec2 model.</p>
          </div>
        </div>

        <button onClick={handleUpload} className="transcribe-btn" disabled={loading}>
          {loading ? "Transcribing..." : "Upload & Transcribe"}
        </button>

        {error && <p className="error-text">{error}</p>}

        {transcription && (
          <div className="transcription-output">
            <h3>📝 Transcription:</h3>
            <p>{transcription}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default SpeechRecognition;
