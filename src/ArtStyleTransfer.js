import React, { useState } from "react";
import axios from "axios";
import "./css/artstyle.css";

function ArtisticStyleTransfer() {
  const [contentImage, setContentImage] = useState(null);
  const [styleImage, setStyleImage] = useState(null);
  const [contentPreview, setContentPreview] = useState(null);
  const [stylePreview, setStylePreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e, type) => {
    const file = e.target.files[0];
    if (!file) return;

    if (type === "content") {
      setContentImage(file);
      setContentPreview(URL.createObjectURL(file));
    }

    if (type === "style") {
      setStyleImage(file);
      setStylePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async () => {
    setError("");
    setResultImage(null);

    if (!contentImage || !styleImage) {
      setError("🎨 Please upload both Content and Style images.");
      return;
    }

    const formData = new FormData();
    formData.append("content_image", contentImage);
    formData.append("style_image", styleImage);

    try {
      setLoading(true);
      const res = await axios.post("http://localhost:5000/artstyle", formData, {
        responseType: "blob",
      });
      const imageUrl = URL.createObjectURL(res.data);
      setResultImage(imageUrl);
    } catch {
      setError("⚠️ Failed to process images. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="artistic-container">
      <h1>🎨 Neural Artistic Style Transfer</h1>
      <p className="subtitle">
        Blend your content photo with an artistic style seamlessly.
      </p>

      <div className="upload-section">
        <div className="upload-card">
          <p>Content Image 📷</p>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => handleChange(e, "content")}
          />
          {contentPreview && (
            <img
              src={contentPreview}
              alt="Content Preview"
              className="preview-image"
            />
          )}
        </div>

        <div className="upload-card">
          <p>Style Image 🖌️</p>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => handleChange(e, "style")}
          />
          {stylePreview && (
            <img
              src={stylePreview}
              alt="Style Preview"
              className="preview-image"
            />
          )}
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="artistic-button"
      >
        {loading ? "✨ Processing..." : "🎨 Transfer Style"}
      </button>

      {error && <p className="artistic-error">{error}</p>}

      {resultImage && (
        <div className="result-section">
          <h3>🖼️ Stylized Image:</h3>
          <img
            src={resultImage}
            alt="Stylized Output"
            className="result-image"
          />
        </div>
      )}
    </div>
  );
}

export default ArtisticStyleTransfer;
