'use client'

import { useState, useEffect } from 'react';
import { useApiEndpoint } from '../hooks/useApiEndpoint'; // make sure path is correct

const ModelToggle = () => {
  const { apiBaseUrl } = useApiEndpoint(); // dynamically get base URL
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch health API to get current model status
  const fetchHealth = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/health`);
      const data = await res.json();
      setModelLoaded(data.model_loaded);
    } catch (err) {
      console.error('Failed to fetch health:', err);
      setModelLoaded(null);
    }
  };

  useEffect(() => {
    fetchHealth();
    // Optional: poll every 10 sec to auto-update status
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]); // re-run if base URL changes

  const handleLoadUnload = async () => {
    if (loading || modelLoaded === null) return;

    setLoading(true);
    try {
      const endpoint = modelLoaded ? 'unload_model' : 'load_model';
      await fetch(`${apiBaseUrl}/${endpoint}`, { method: 'POST' });
      // Refresh status after a delay
      setTimeout(fetchHealth, 5000);
    } catch (err) {
      console.error('Error in load/unload:', err);
    } finally {
      setTimeout(() => setLoading(false), 5000);
    }
  };

  return (
    <button
      onClick={handleLoadUnload}
      disabled={loading || modelLoaded === null}
      className={`p-2 rounded-lg border bg-card border-border cursor-pointer duration-300 ${
        loading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-accent'
      }`}
      title={modelLoaded ? 'Unload model' : 'Load model'}
    >
      {loading
        ? 'Processing...'
        : modelLoaded
        ? 'Unload Model'
        : 'Load Model'}
    </button>
  );
};

export default ModelToggle;
