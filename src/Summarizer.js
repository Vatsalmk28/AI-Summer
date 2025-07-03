// src/Summarizer.js
import React, { useState } from "react";
import axios from "axios";
import "./css/summarizer.css";

function Summarizer() {
  const [inputType, setInputType] = useState("url");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError("");
    setLoading(true);

    try {
      let response;
      if (inputType === "url") {
        response = await axios.post("http://localhost:5000/summarize", { url });
      } else if (inputType === "text") {
        response = await axios.post("http://localhost:5000/summarize", { text });
      } else if (inputType === "pdf") {
        const formData = new FormData();
        formData.append("file", file);
        response = await axios.post("http://localhost:5000/summarize", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      }
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="title">🧠 Smart Summarizer</h1>
      <div className="cards-container">
        <div className={`card ${inputType === "url" ? "active" : ""}`} onClick={() => setInputType("url")}>
          <h2>🔗 URL</h2><p>Paste an article link.</p>
        </div>
        <div className={`card ${inputType === "pdf" ? "active" : ""}`} onClick={() => setInputType("pdf")}>
          <h2>📄 PDF</h2><p>Upload a PDF document.</p>
        </div>
        <div className={`card ${inputType === "text" ? "active" : ""}`} onClick={() => setInputType("text")}>
          <h2>✍️ Text</h2><p>Paste or type raw text.</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="form">
        {inputType === "url" && (
          <input type="text" placeholder="Enter article URL" value={url} onChange={(e) => setUrl(e.target.value)} required className="input" />
        )}
        {inputType === "text" && (
          <textarea placeholder="Paste your text here..." value={text} onChange={(e) => setText(e.target.value)} rows={8} required className="textarea" />
        )}
        {inputType === "pdf" && (
          <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])} required className="input" />
        )}
        <button type="submit" disabled={loading} className="button">{loading ? "Summarizing..." : "Submit"}</button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result">
          <h3>📝 Summary</h3>
          <p><strong>Title:</strong> {result.title}</p>
          <p><strong>Authors:</strong> {result.authors?.join(", ")}</p>
          <p><strong>Published on:</strong> {result.publish_date}</p>
          <p className="summary-text"><strong>Summary:</strong> {result.summary}</p>
        </div>
      )}
    </div>
  );
}

export default Summarizer;
