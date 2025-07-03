// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Summarizer from "./Summarizer";
import SpeechRecognition from "./SpeechRecognition";
import ArtStyleTransfer from "./ArtStyleTransfer";
import Home from "./Home";
import "./App.css";
import TextGen from "./TextGen";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/summarizer" element={<Summarizer />} />
        <Route path="/speech" element={<SpeechRecognition />} />
        <Route path="/artstyle" element={<ArtStyleTransfer />} />
        <Route path="/textgen" element={<TextGen />} />
      </Routes>
    </Router>
  );
}

export default App;
