import axios from 'axios';

// TODO: Replace with the actual deployed backend API URL
const API_BASE_URL = 'http://localhost:8000'; // Placeholder URL

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictEnergy = async (timestamps, meter_ids, context_features) => {
  try {
    const response = await apiClient.post('/predict', {
      timestamps,
      meter_ids,
      context_features,
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error.response ? error.response.data : error.message);
    // Rethrow a more specific error or return a structured error object
    throw new Error(error.response?.data?.detail || 'Failed to fetch prediction');
  }
};

export default apiClient;

