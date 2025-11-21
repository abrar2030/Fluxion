import React, { createContext, useContext, useState, useEffect } from "react";
import { marketAPI, poolsAPI, analyticsAPI } from "../services/api";

// Create Data Context
const DataContext = createContext(null);

export const DataProvider = ({ children }) => {
  const [marketData, setMarketData] = useState(null);
  const [poolsData, setPoolsData] = useState([]);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch market data
  const fetchMarketData = async () => {
    try {
      setIsLoading(true);

      // Try to fetch from API, fall back to mock data if it fails
      try {
        const response = await marketAPI.getMarketData();
        setMarketData(response.data);
      } catch (err) {
        console.warn("Using mock market data due to API error:", err);
        // Mock data as fallback
        setMarketData({
          tvl: "$8.2M",
          volume24h: "$840K",
          activePools: 24,
          avgApy: "5.8%",
        });
      }

      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching market data:", error);
      setError("Failed to fetch market data. Please try again.");
      setIsLoading(false);
    }
  };

  // Fetch pools data
  const fetchPoolsData = async () => {
    try {
      setIsLoading(true);

      // Try to fetch from API, fall back to mock data if it fails
      try {
        const response = await poolsAPI.getAllPools();
        setPoolsData(response.data);
      } catch (err) {
        console.warn("Using mock pools data due to API error:", err);
        // Mock data as fallback
        setPoolsData([
          {
            id: "ETH-USDC",
            tvl: "$2.4M",
            volume24h: "$340K",
            apy: "5.2%",
            risk: "Low",
            assets: ["ETH", "USDC"],
            weights: [50, 50],
            utilization: 78,
          },
          {
            id: "BTC-ETH",
            tvl: "$1.8M",
            volume24h: "$220K",
            apy: "4.8%",
            risk: "Medium",
            assets: ["BTC", "ETH"],
            weights: [60, 40],
            utilization: 65,
          },
          {
            id: "LINK-ETH",
            tvl: "$950K",
            volume24h: "$120K",
            apy: "7.3%",
            risk: "Medium",
            assets: ["LINK", "ETH"],
            weights: [30, 70],
            utilization: 82,
          },
          {
            id: "UNI-USDT",
            tvl: "$750K",
            volume24h: "$85K",
            apy: "6.1%",
            risk: "Low",
            assets: ["UNI", "USDT"],
            weights: [40, 60],
            utilization: 71,
          },
          {
            id: "AAVE-WBTC",
            tvl: "$1.2M",
            volume24h: "$180K",
            apy: "5.8%",
            risk: "Medium",
            assets: ["AAVE", "WBTC"],
            weights: [35, 65],
            utilization: 68,
          },
          {
            id: "SNX-ETH",
            tvl: "$680K",
            volume24h: "$92K",
            apy: "8.2%",
            risk: "High",
            assets: ["SNX", "ETH"],
            weights: [25, 75],
            utilization: 89,
          },
        ]);
      }

      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching pools data:", error);
      setError("Failed to fetch pools data. Please try again.");
      setIsLoading(false);
    }
  };

  // Fetch analytics data
  const fetchAnalyticsData = async () => {
    try {
      setIsLoading(true);

      // Try to fetch from API, fall back to mock data if it fails
      try {
        const volumeResponse = await analyticsAPI.getVolumeData();
        const tvlResponse = await analyticsAPI.getTvlData();
        const distributionResponse = await analyticsAPI.getPoolDistribution();

        setAnalyticsData({
          volumeData: volumeResponse.data,
          tvlData: tvlResponse.data,
          poolDistribution: distributionResponse.data,
        });
      } catch (err) {
        console.warn("Using mock analytics data due to API error:", err);
        // Mock data as fallback
        const mockVolumeData = [
          { name: "Jan", volume: 2400 },
          { name: "Feb", volume: 1398 },
          { name: "Mar", volume: 9800 },
          { name: "Apr", volume: 3908 },
          { name: "May", volume: 4800 },
          { name: "Jun", volume: 3800 },
          { name: "Jul", volume: 4300 },
        ];

        const mockTvlData = [
          { name: "Jan", tvl: 4000 },
          { name: "Feb", tvl: 3000 },
          { name: "Mar", tvl: 2000 },
          { name: "Apr", tvl: 2780 },
          { name: "May", tvl: 1890 },
          { name: "Jun", tvl: 2390 },
          { name: "Jul", tvl: 3490 },
        ];

        const mockPoolDistribution = [
          { name: "ETH-USDC", value: 2400 },
          { name: "BTC-ETH", value: 1800 },
          { name: "LINK-ETH", value: 950 },
          { name: "UNI-USDT", value: 750 },
          { name: "Others", value: 1200 },
        ];

        setAnalyticsData({
          volumeData: mockVolumeData,
          tvlData: mockTvlData,
          poolDistribution: mockPoolDistribution,
        });
      }

      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching analytics data:", error);
      setError("Failed to fetch analytics data. Please try again.");
      setIsLoading(false);
    }
  };

  // Initialize data
  useEffect(() => {
    fetchMarketData();
    fetchPoolsData();
    fetchAnalyticsData();
  }, []);

  // Context value
  const value = {
    marketData,
    poolsData,
    analyticsData,
    isLoading,
    error,
    fetchMarketData,
    fetchPoolsData,
    fetchAnalyticsData,
  };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
};

// Custom hook to use the Data context
export const useData = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error("useData must be used within a DataProvider");
  }
  return context;
};

export default useData;
