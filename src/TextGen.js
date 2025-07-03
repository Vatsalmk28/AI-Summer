import React, { useState } from "react";
import axios from "axios";
import "./css/textgen.css";

function TextGenerator() {
  const [topic, setTopic] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleGenerate = async () => {
    setError("");
    setResult("");
    if (!topic.trim()) {
      setError("Please enter a topic.");
      return;
    }

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/textgen", { topic });
      setResult(res.data.generated_text);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="textgen-container">
      <h1>📝 Text Generator</h1>
      <p className="subtitle">
        Enter a topic and generate a meaningful paragraph about it.
      </p>

      <div className="cards">
        <div className="card">
          <p><strong>Topic Name</strong> 📝</p>
          <input
            type="text"
            placeholder="Enter your topic..."
            value={topic}
            id="topic-input"
            onChange={(e) => setTopic(e.target.value)}
          />
        </div>
      </div>

      <button
        onClick={handleGenerate}
        disabled={loading}
        className="generate-btn"
      >
        {loading ? "Generating..." : "✨ Generate Text"}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result-box">
          <h3>🧾 Generated Text:</h3>
          <p className="generated-text">{result}</p>
        </div>
      )}
    </div>
  );
}

export default TextGenerator;
