const API_URL = "http://localhost:8000";

export interface PredictionResponse {
  predicted_rul: number;
}

export const predictRUL = async (features: number[]): Promise<number> => {
  if (features.length !== 24) {
    throw new Error(`Expected 24 inputs, got ${features.length}`);
  }

  // The LSTM expects a sequence window of 30 time steps. 
  // We duplicate the 24-feature array 30 times to match the tensor shape.
  const sequenceLength = 30;
  const windowData = Array(sequenceLength).fill(features);

  const payload = {
    window_data: windowData
  };

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    // If FastAPI throws a 400 or 422, this captures the exact Pydantic error details
    const errorData = await response.json().catch(() => null);
    const errorMessage = errorData ? JSON.stringify(errorData.detail || errorData) : response.statusText;
    throw new Error(`Backend Rejected Payload: ${errorMessage}`);
  }

  const data: PredictionResponse = await response.json();
  return data.predicted_rul;
};