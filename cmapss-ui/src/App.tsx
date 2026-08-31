import { useState } from 'react';
import { predictRUL } from './api';

function App() {
  // 3 Operational Settings + 21 Sensors = 24 Features
  const [features, setFeatures] = useState<number[]>([
    0.0, 0.0, 100.0, // Settings 1-3
    518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30,
    47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190 // Sensors 1-21
  ]);
  
  const [rul, setRul] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFeatureChange = (index: number, value: string) => {
    const newFeatures = [...features];
    newFeatures[index] = parseFloat(value) || 0;
    setFeatures(newFeatures);
  };

  const executePrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictRUL(features);
      setRul(result);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-mono">
      <div className="max-w-4xl mx-auto">
        
        <header className="mb-8 border-b border-gray-800 pb-4">
          <h1 className="text-2xl font-bold text-blue-400">Turbofan RUL Inference Terminal</h1>
          <p className="text-gray-500 text-sm mt-1">Model: cmapss-baseline-lstm | Status: Online</p>
        </header>

        <main className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          <div className="col-span-2 space-y-6">
            {/* Operational Settings */}
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-300">Operational Settings</h2>
              <div className="grid grid-cols-3 gap-4">
                {features.slice(0, 3).map((value, idx) => (
                  <div key={`op-${idx}`}>
                    <label className="block text-xs text-gray-500 mb-1">Op Set {idx + 1}</label>
                    <input
                      type="number"
                      step="any"
                      className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500 transition-colors"
                      value={value}
                      onChange={(e) => handleFeatureChange(idx, e.target.value)}
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Sensor Array */}
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <h2 className="text-lg font-semibold mb-4 text-gray-300">Sensor Array (21 Channels)</h2>
              <div className="grid grid-cols-4 gap-4">
                {features.slice(3).map((value, idx) => (
                  <div key={`sensor-${idx}`}>
                    <label className="block text-xs text-gray-500 mb-1">S-{idx + 1}</label>
                    <input
                      type="number"
                      step="any"
                      className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500 transition-colors"
                      value={value}
                      onChange={(e) => handleFeatureChange(idx + 3, e.target.value)}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Execution & Output Panel */}
          <div className="col-span-1 flex flex-col gap-6">
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg flex flex-col items-center justify-center h-full">
              <button
                onClick={executePrediction}
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded transition-colors disabled:opacity-50"
              >
                {loading ? 'Executing...' : 'INITIATE INFERENCE'}
              </button>

              {error && (
                <div className="mt-4 p-3 w-full bg-red-900/50 border border-red-500 text-red-200 text-xs rounded break-words">
                  {error}
                </div>
              )}

              {rul !== null && !error && (
                <div className="mt-8 text-center w-full">
                  <span className="block text-gray-400 text-sm mb-2">Predicted RUL</span>
                  <span className={`text-5xl font-bold ${rul > 50 ? 'text-green-400' : 'text-red-400'}`}>
                    {rul.toFixed(2)}
                  </span>
                  <span className="block text-gray-500 text-xs mt-2">cycles remaining</span>
                </div>
              )}
            </div>
          </div>

        </main>
      </div>
    </div>
  );
}

export default App;