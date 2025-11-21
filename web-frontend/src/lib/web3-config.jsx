import { createContext, useContext, useState, useEffect } from "react";
import { ethers } from "ethers";
import { userAPI } from "../services/api";

// ABIs
const POOL_MANAGER_ABI = [
  // Add Pool Manager ABI here
  {
    anonymous: false,
    inputs: [
      {
        indexed: true,
        internalType: "bytes32",
        name: "poolId",
        type: "bytes32",
      },
      {
        indexed: false,
        internalType: "address[]",
        name: "assets",
        type: "address[]",
      },
      {
        indexed: false,
        internalType: "uint256[]",
        name: "weights",
        type: "uint256[]",
      },
    ],
    name: "PoolCreated",
    type: "event",
  },
];

const FACTORY_ABI = [
  // Add Factory ABI here
  {
    inputs: [
      {
        internalType: "address[]",
        name: "assets",
        type: "address[]",
      },
      {
        internalType: "uint256[]",
        name: "weights",
        type: "uint256[]",
      },
      {
        internalType: "uint256",
        name: "swapFee",
        type: "uint256",
      },
      {
        internalType: "uint256",
        name: "amplificationParameter",
        type: "uint256",
      },
    ],
    name: "createPool",
    outputs: [
      {
        internalType: "bytes32",
        name: "",
        type: "bytes32",
      },
    ],
    stateMutability: "nonpayable",
    type: "function",
  },
];

// Create Web3 Context
const Web3Context = createContext(null);

export function Web3Provider({ children }) {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [account, setAccount] = useState(null);
  const [contracts, setContracts] = useState({});
  const [pools, setPools] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [chainId, setChainId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Initialize provider
  useEffect(() => {
    const init = async () => {
      setIsLoading(true);
      setError(null);

      // Check if window.ethereum is available
      if (window.ethereum) {
        try {
          // Fix for ethers v6
          const provider = new ethers.BrowserProvider(window.ethereum);
          setProvider(provider);

          // Get network
          const network = await provider.getNetwork();
          setChainId(network.chainId);

          // Initialize contracts
          initializeContracts(provider);

          // Listen for account changes
          window.ethereum.on("accountsChanged", handleAccountsChanged);

          // Listen for chain changes
          window.ethereum.on("chainChanged", () => window.location.reload());

          // Check if already connected
          const accounts = await provider.listAccounts();
          if (accounts.length > 0) {
            handleAccountsChanged(accounts);
          }

          setIsLoading(false);
        } catch (error) {
          console.error("Error initializing web3:", error);
          setError(
            "Failed to initialize Web3. Please make sure you have a compatible wallet installed.",
          );
          setIsLoading(false);
        }
      } else {
        setError(
          "No Ethereum wallet detected. Please install MetaMask or another compatible wallet.",
        );
        setIsLoading(false);
      }
    };

    init();

    return () => {
      if (window.ethereum) {
        window.ethereum.removeListener(
          "accountsChanged",
          handleAccountsChanged,
        );
      }
    };
  }, []);

  // Initialize contracts
  const initializeContracts = (provider) => {
    try {
      const poolManagerAddress =
        import.meta.env.VITE_POOL_MANAGER_ADDRESS ||
        "0x0000000000000000000000000000000000000000";
      const factoryAddress =
        import.meta.env.VITE_FACTORY_ADDRESS ||
        "0x0000000000000000000000000000000000000000";

      const poolManager = new ethers.Contract(
        poolManagerAddress,
        POOL_MANAGER_ABI,
        provider,
      );

      const factory = new ethers.Contract(
        factoryAddress,
        FACTORY_ABI,
        provider,
      );

      setContracts({ poolManager, factory });
    } catch (error) {
      console.error("Error initializing contracts:", error);
      setError(
        "Failed to initialize smart contracts. Please check your network connection.",
      );
    }
  };

  // Handle account changes
  const handleAccountsChanged = async (accounts) => {
    if (accounts.length === 0) {
      // User disconnected
      setAccount(null);
      setSigner(null);
      setIsConnected(false);
      setPools([]);
    } else {
      // User connected or changed account
      setAccount(accounts[0]);

      try {
        // Fix for ethers v6
        const signer = await provider.getSigner();
        setSigner(signer);
        setIsConnected(true);

        // Fetch pools for the connected account
        fetchPools(accounts[0]);

        // Update user profile on backend
        try {
          await userAPI.updateProfile({
            address: accounts[0],
            chainId: chainId,
          });
        } catch (err) {
          console.warn("Failed to update user profile:", err);
        }
      } catch (error) {
        console.error("Error setting signer:", error);
        setError("Failed to connect to your wallet. Please try again.");
      }
    }
  };

  // Connect wallet
  const connectWallet = async () => {
    setIsLoading(true);
    setError(null);

    if (provider) {
      try {
        const accounts = await window.ethereum.request({
          method: "eth_requestAccounts",
        });
        handleAccountsChanged(accounts);
        setIsLoading(false);
      } catch (error) {
        console.error("Error connecting wallet:", error);
        setError(
          "Failed to connect wallet. Please check your wallet and try again.",
        );
        setIsLoading(false);
      }
    } else {
      setError(
        "No Ethereum provider available. Please install MetaMask or another compatible wallet.",
      );
      setIsLoading(false);
    }
  };

  // Fetch pools
  const fetchPools = async (address) => {
    if (contracts.poolManager && address) {
      setIsLoading(true);

      try {
        // Mock data for development
        const mockPools = [
          {
            id: "pool-" + address.substring(0, 8) + "-1",
            assets: ["ETH", "USDC"],
            weights: [50, 50],
            fee: 0.3,
            amplification: 100,
            tvl: "$1.2M",
            apy: "4.5%",
          },
          {
            id: "pool-" + address.substring(0, 8) + "-2",
            assets: ["ETH", "WBTC"],
            weights: [40, 60],
            fee: 0.25,
            amplification: 85,
            tvl: "$850K",
            apy: "5.2%",
          },
        ];

        setPools(mockPools);
        setIsLoading(false);
      } catch (error) {
        console.error("Error fetching pools:", error);
        setError("Failed to fetch your pools. Please try again later.");
        setIsLoading(false);
      }
    }
  };

  // Create a new pool
  const createPool = async (poolData) => {
    if (!isConnected || !signer || !contracts.factory) {
      setError("Please connect your wallet first");
      return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Simulate pool creation
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Mock new pool
      const newPool = {
        id: "pool-" + account.substring(0, 8) + "-" + (pools.length + 1),
        assets: poolData.assets,
        weights: poolData.weights,
        fee: poolData.fee,
        amplification: poolData.amplification,
        tvl: "$0",
        apy: "0%",
      };

      setPools([...pools, newPool]);
      setIsLoading(false);

      return newPool.id;
    } catch (error) {
      console.error("Error creating pool:", error);
      setError(
        "Failed to create pool. Please check your inputs and try again.",
      );
      setIsLoading(false);
      return null;
    }
  };

  // Context value
  const value = {
    provider,
    signer,
    account,
    contracts,
    pools,
    isConnected,
    chainId,
    isLoading,
    error,
    connectWallet,
    fetchPools,
    createPool,
  };

  return <Web3Context.Provider value={value}>{children}</Web3Context.Provider>;
}

// Custom hook to use the Web3 context
export function useWeb3() {
  const context = useContext(Web3Context);
  if (!context) {
    throw new Error("useWeb3 must be used within a Web3Provider");
  }
  return context;
}
