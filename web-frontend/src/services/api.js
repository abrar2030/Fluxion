import axios from "axios";

// Create an axios instance with default config
const api = axios.create({
  baseURL: "/api",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Add request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("fluxion_token");
    if (token) {
      config.headers["Authorization"] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      if (error.response.status === 401) {
        // Unauthorized - clear token and redirect to login
        localStorage.removeItem("fluxion_token");
        // Could dispatch an action or trigger a notification here
      }
    } else if (error.request) {
      // Request made but no response received
      console.error("Network error, no response received");
    } else {
      // Error in setting up the request
      console.error("Error setting up request:", error.message);
    }
    return Promise.reject(error);
  },
);

// API endpoints
export const marketAPI = {
  getMarketData: () => api.get("/market/data"),
  getTokenPrices: () => api.get("/market/prices"),
  getHistoricalData: (params) => api.get("/market/historical", { params }),
};

export const poolsAPI = {
  getAllPools: () => api.get("/pools"),
  getPoolById: (id) => api.get(`/pools/${id}`),
  createPool: (data) => api.post("/pools", data),
  addLiquidity: (id, data) => api.post(`/pools/${id}/liquidity`, data),
  removeLiquidity: (id, data) => api.post(`/pools/${id}/withdraw`, data),
};

export const analyticsAPI = {
  getTVL: () => api.get("/analytics/tvl"),
  getVolume: () => api.get("/analytics/volume"),
  getPoolDistribution: () => api.get("/analytics/distribution"),
  getUserStats: () => api.get("/analytics/user"),
};

export const userAPI = {
  login: (data) => api.post("/auth/login", data),
  register: (data) => api.post("/auth/register", data),
  getProfile: () => api.get("/user/profile"),
  updateProfile: (data) => api.put("/user/profile", data),
  getPositions: () => api.get("/user/positions"),
};

export const syntheticsAPI = {
  getAllSynthetics: () => api.get("/synthetics"),
  getSyntheticById: (id) => api.get(`/synthetics/${id}`),
  mintSynthetic: (data) => api.post("/synthetics/mint", data),
  burnSynthetic: (data) => api.post("/synthetics/burn", data),
  getUserPositions: () => api.get("/synthetics/positions"),
};

// Fallback to mock data when API is not available
const mockData = {
  market: {
    tvl: "$142.5M",
    volume24h: "$28.4M",
    activePools: 247,
    averageApy: "8.74%",
  },
  pools: [
    {
      id: "pool-1",
      name: "ETH-USDC",
      assets: ["ETH", "USDC"],
      weights: [50, 50],
      tvl: "$4.2M",
      volume24h: "$1.2M",
      apy: "8.4%",
      fee: "0.3%",
    },
    {
      id: "pool-2",
      name: "WBTC-ETH",
      assets: ["WBTC", "ETH"],
      weights: [60, 40],
      tvl: "$8.7M",
      volume24h: "$3.5M",
      apy: "7.2%",
      fee: "0.3%",
    },
  ],
};

// Helper function to get mock data when API fails
export const getMockData = (type) => {
  return mockData[type] || null;
};

export default api;
