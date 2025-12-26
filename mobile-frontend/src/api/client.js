import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';

// Get API configuration from environment or use defaults
const getApiBaseUrl = () => {
    // Try to get from expo constants first (for managed workflow)
    const expoConfig = Constants.expoConfig || Constants.manifest;
    if (expoConfig?.extra?.apiBaseUrl) {
        return expoConfig.extra.apiBaseUrl;
    }

    // Fallback to default development URL
    return __DEV__ ? 'http://localhost:8000' : 'https://api.fluxion.io';
};

const API_BASE_URL = getApiBaseUrl();
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: API_TIMEOUT,
});

// Token storage keys
const AUTH_TOKEN_KEY = '@fluxion_auth_token';
const TOKEN_EXPIRY_KEY = '@fluxion_token_expiry';

// Token management functions
export const setAuthToken = async (token, expiryMinutes = 30) => {
    try {
        await AsyncStorage.setItem(AUTH_TOKEN_KEY, token);
        const expiryTime = Date.now() + expiryMinutes * 60 * 1000;
        await AsyncStorage.setItem(TOKEN_EXPIRY_KEY, expiryTime.toString());
    } catch (error) {
        console.error('Error saving auth token:', error);
    }
};

export const getAuthToken = async () => {
    try {
        const token = await AsyncStorage.getItem(AUTH_TOKEN_KEY);
        const expiry = await AsyncStorage.getItem(TOKEN_EXPIRY_KEY);

        if (!token || !expiry) {
            return null;
        }

        // Check if token is expired
        if (Date.now() > parseInt(expiry, 10)) {
            await clearAuthToken();
            return null;
        }

        return token;
    } catch (error) {
        console.error('Error getting auth token:', error);
        return null;
    }
};

export const clearAuthToken = async () => {
    try {
        await AsyncStorage.multiRemove([AUTH_TOKEN_KEY, TOKEN_EXPIRY_KEY]);
    } catch (error) {
        console.error('Error clearing auth token:', error);
    }
};

// Request interceptor for adding auth token
apiClient.interceptors.request.use(
    async (config) => {
        const token = await getAuthToken();
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error),
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
    (response) => response,
    async (error) => {
        // Handle session expiration
        if (error.response && error.response.status === 401) {
            await clearAuthToken();
            // You can dispatch a navigation event here if needed
            // For now, we just clear the token
        }

        // Handle network errors
        if (!error.response) {
            error.message = 'Network error. Please check your connection.';
        }

        return Promise.reject(error);
    },
);

// Retry logic for failed requests
const retryRequest = async (requestFn, retries = 3, delay = 1000) => {
    for (let i = 0; i < retries; i++) {
        try {
            return await requestFn();
        } catch (error) {
            if (i === retries - 1) throw error;
            await new Promise((resolve) => setTimeout(resolve, delay * (i + 1)));
        }
    }
};

// API Functions

/**
 * Predict energy consumption
 * @param {Array<string>} timestamps - ISO 8601 formatted timestamps
 * @param {Array<string>} meterIds - Meter IDs
 * @param {Object} contextFeatures - Context features object
 * @returns {Promise<Object>} Prediction results
 */
export const predictEnergy = async (timestamps, meterIds, contextFeatures) => {
    try {
        const response = await retryRequest(() =>
            apiClient.post('/predict', {
                timestamps,
                meter_ids: meterIds,
                context_features: contextFeatures,
            }),
        );
        return response.data;
    } catch (error) {
        console.error('API Error (predictEnergy):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch prediction');
    }
};

/**
 * Fetch all synthetic assets
 * @param {Object} params - Query parameters (limit, offset, search, etc.)
 * @returns {Promise<Object>} Assets data
 */
export const fetchAssets = async (params = {}) => {
    try {
        const response = await retryRequest(() => apiClient.get('/assets', { params }));
        return response.data;
    } catch (error) {
        console.error('API Error (fetchAssets):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch assets');
    }
};

/**
 * Fetch single asset by ID
 * @param {string} assetId - Asset ID
 * @returns {Promise<Object>} Asset data
 */
export const fetchAssetById = async (assetId) => {
    try {
        const response = await retryRequest(() => apiClient.get(`/assets/${assetId}`));
        return response.data;
    } catch (error) {
        console.error('API Error (fetchAssetById):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch asset data');
    }
};

/**
 * Fetch all liquidity pools
 * @param {Object} params - Query parameters (limit, offset, search, etc.)
 * @returns {Promise<Object>} Pools data
 */
export const fetchPools = async (params = {}) => {
    try {
        const response = await retryRequest(() => apiClient.get('/pools', { params }));
        return response.data;
    } catch (error) {
        console.error('API Error (fetchPools):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch pools');
    }
};

/**
 * Fetch single pool by ID
 * @param {string} poolId - Pool ID
 * @returns {Promise<Object>} Pool data
 */
export const fetchPoolById = async (poolId) => {
    try {
        const response = await retryRequest(() => apiClient.get(`/pools/${poolId}`));
        return response.data;
    } catch (error) {
        console.error('API Error (fetchPoolById):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch pool data');
    }
};

/**
 * Fetch asset price history
 * @param {string} assetId - Asset ID
 * @param {string} timeframe - Timeframe (1h, 24h, 7d, 30d, 1y)
 * @returns {Promise<Object>} Price history data
 */
export const fetchAssetPriceHistory = async (assetId, timeframe = '24h') => {
    try {
        const response = await retryRequest(() =>
            apiClient.get(`/assets/${assetId}/price-history`, {
                params: { timeframe },
            }),
        );
        return response.data;
    } catch (error) {
        console.error('API Error (fetchAssetPriceHistory):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch price history');
    }
};

/**
 * Create a new trade order
 * @param {Object} orderData - Order data (asset_id, type, amount, price, etc.)
 * @returns {Promise<Object>} Order confirmation
 */
export const createOrder = async (orderData) => {
    try {
        const response = await apiClient.post('/orders', orderData);
        return response.data;
    } catch (error) {
        console.error('API Error (createOrder):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to create order');
    }
};

/**
 * Fetch user's orders
 * @param {Object} params - Query parameters (status, limit, offset)
 * @returns {Promise<Object>} Orders data
 */
export const fetchUserOrders = async (params = {}) => {
    try {
        const response = await retryRequest(() => apiClient.get('/orders', { params }));
        return response.data;
    } catch (error) {
        console.error('API Error (fetchUserOrders):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch orders');
    }
};

/**
 * Fetch user's portfolio
 * @returns {Promise<Object>} Portfolio data
 */
export const fetchPortfolio = async () => {
    try {
        const response = await retryRequest(() => apiClient.get('/portfolio'));
        return response.data;
    } catch (error) {
        console.error('API Error (fetchPortfolio):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Failed to fetch portfolio');
    }
};

/**
 * Authenticate user
 * @param {string} address - Wallet address
 * @param {string} signature - Signed message
 * @returns {Promise<Object>} Authentication response with token
 */
export const authenticateUser = async (address, signature) => {
    try {
        const response = await apiClient.post('/auth/login', {
            address,
            signature,
        });

        // Store the token
        if (response.data.access_token) {
            await setAuthToken(response.data.access_token);
        }

        return response.data;
    } catch (error) {
        console.error('API Error (authenticateUser):', error.response?.data || error.message);
        throw new Error(error.response?.data?.detail || 'Authentication failed');
    }
};

/**
 * Get health status of the API
 * @returns {Promise<Object>} Health status
 */
export const getHealthStatus = async () => {
    try {
        const response = await apiClient.get('/health');
        return response.data;
    } catch (error) {
        console.error('API Error (getHealthStatus):', error.response?.data || error.message);
        throw new Error('API health check failed');
    }
};

export default apiClient;
