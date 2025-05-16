import axios from 'axios';

// Production API URL for the Fluxion backend
const API_BASE_URL = 'https://api.fluxion.io/v1'; 

// Fallback to localhost for development environment
const isDevelopment = process.env.NODE_ENV === 'development';
const apiBaseUrl = isDevelopment ? 'http://localhost:8000' : API_BASE_URL;

const apiClient = axios.create({
  baseURL: apiBaseUrl,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout
});

// Request interceptor for adding auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('fluxion_auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle session expiration
    if (error.response && error.response.status === 401) {
      // Clear invalid token
      localStorage.removeItem('fluxion_auth_token');
      // Redirect to login if needed
      if (typeof window !== 'undefined') {
        // Check if we're not in a test environment
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

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
    // Return structured error object
    throw new Error(error.response?.data?.detail || 'Failed to fetch prediction');
  }
};

export const fetchPoolData = async (poolId) => {
  try {
    const response = await apiClient.get(`/pools/${poolId}`);
    return response.data;
  } catch (error) {
    console.error('API Error:', error.response ? error.response.data : error.message);
    throw new Error(error.response?.data?.detail || 'Failed to fetch pool data');
  }
};

export const fetchAssetData = async (assetId) => {
  try {
    const response = await apiClient.get(`/assets/${assetId}`);
    return response.data;
  } catch (error) {
    console.error('API Error:', error.response ? error.response.data : error.message);
    throw new Error(error.response?.data?.detail || 'Failed to fetch asset data');
  }
};

export default apiClient;
