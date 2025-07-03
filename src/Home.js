// src/Home.js
import React from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <h1 className="home-title">Welcome to AI Assistant</h1>
      <p className="home-subtitle">
        🚀 Experience intelligent content summarization and speech-to-text
        conversion powered by AI.
      </p>

      <div className="feature-boxes">
        <div className="feature-card">
          <h2>📰 Article Summarizer</h2>
          <p>
            Automatically summarize lengthy news articles, PDF documents, or raw
            text into clear, concise highlights.
          </p>
          <button onClick={() => navigate("/summarizer")}>
            Go to Summarizer
          </button>
        </div>

        <div className="feature-card">
          <h2>🎤 Speech Recognizer</h2>
          <p>
            Upload an audio file (.mp3, .wav, .m4a) and convert speech into
            readable text using Whisper or Wav2Vec2.
          </p>
          <button onClick={() => navigate("/speech")}>
            Go to Speech Recognition
          </button>
        </div>
        <div className="feature-card">
          <h2>🎨 Artistic Style Transfer</h2>
          <p>
            Blend the content of one image with the artistic style of another to
            create beautiful, stylized artwork powered by deep learning.
          </p>
          <button onClick={() => navigate("/artstyle")}>
            Go to Style Transfer
          </button>
        </div>
        <div className="feature-card">
          <h2>📝 Text Generator</h2>
          <p>
            Enter a topic name and generate a long, meaningful paragraph about
            it using advanced language models.
          </p>
          <button onClick={() => navigate("/textgen")}>
            Go to Text Generation
          </button>
        </div>
      </div>
    </div>
  );
}

export default Home;
