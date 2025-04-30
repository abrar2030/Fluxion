import axios from 'axios';

// Assuming the backend runs locally during development
// TODO: Replace with actual backend URL when deployed or configured
const API_URL = 'http://localhost:5000'; // Default Flask port

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Example API call function (adjust based on actual backend endpoints)
export const getHealth = () => {
  return apiClient.get('/health');
};

export const getPools = async () => {
  // Replace with actual endpoint if backend provides pool data via API
  console.warn('getPools API endpoint not implemented yet.');
  // Example structure:
  // return apiClient.get('/pools');
  return Promise.resolve({ data: [] }); // Placeholder
};

export const getSynthetics = async () => {
  // Replace with actual endpoint if backend provides synthetics data via API
  console.warn('getSynthetics API endpoint not implemented yet.');
  // Example structure:
  // return apiClient.get('/synthetics');
  return Promise.resolve({ data: [] }); // Placeholder
};

// Add other API functions as needed

export default apiClient;

