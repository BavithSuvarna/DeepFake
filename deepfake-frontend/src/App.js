// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/predict', formData);
      setResult(response.data.result);
    } catch (error) {
      setResult("An error occurred while uploading the video.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Deepfake Video Detector</h1>
      <form onSubmit={handleUpload}>
        <input type="file" accept="video/*" onChange={handleFileChange} required />
        <button type="submit">Upload and Predict</button>
      </form>
      {loading && <p>Processing video...</p>}
      {result && (
        <div className="result">
          <h2>Prediction Result:</h2>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
}

export default App;
