import React, { createContext, useContext, useState, useEffect } from 'react';
import { ethers } from 'ethers';

// ABIs
const POOL_MANAGER_ABI = [
  // Add Pool Manager ABI here
];

const FACTORY_ABI = [
  // Add Factory ABI here
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

  // Initialize provider
  useEffect(() => {
    const init = async () => {
      // Check if window.ethereum is available
      if (window.ethereum) {
        try {
          const provider = new ethers.providers.Web3Provider(window.ethereum);
          setProvider(provider);

          // Get network
          const network = await provider.getNetwork();
          setChainId(network.chainId);

          // Initialize contracts
          initializeContracts(provider);

          // Listen for account changes
          window.ethereum.on('accountsChanged', handleAccountsChanged);

          // Listen for chain changes
          window.ethereum.on('chainChanged', () => window.location.reload());

        } catch (error) {
          console.error("Error initializing web3:", error);
        }
      }
    };

    init();

    return () => {
      if (window.ethereum) {
        window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
      }
    };
  }, []);

  // Initialize contracts
  const initializeContracts = (provider) => {
    const poolManager = new ethers.Contract(
      process.env.POOL_MANAGER_ADDRESS || '0x0000000000000000000000000000000000000000',
      POOL_MANAGER_ABI,
      provider
    );

    const factory = new ethers.Contract(
      process.env.FACTORY_ADDRESS || '0x0000000000000000000000000000000000000000',
      FACTORY_ABI,
      provider
    );

    setContracts({ poolManager, factory });
  };

  // Handle account changes
  const handleAccountsChanged = async (accounts) => {
    if (accounts.length === 0) {
      // User disconnected
      setAccount(null);
      setSigner(null);
      setIsConnected(false);
    } else {
      // User connected or changed account
      setAccount(accounts[0]);
      const signer = provider.getSigner();
      setSigner(signer);
      setIsConnected(true);

      // Fetch pools for the connected account
      fetchPools(accounts[0]);
    }
  };

  // Connect wallet
  const connectWallet = async () => {
    if (provider) {
      try {
        const accounts = await window.ethereum.request({
          method: 'eth_requestAccounts'
        });
        handleAccountsChanged(accounts);
      } catch (error) {
        console.error("Error connecting wallet:", error);
      }
    }
  };

  // Fetch pools
  const fetchPools = async (address) => {
    if (contracts.poolManager && address) {
      try {
        // Example implementation - adjust based on actual contract methods
        const poolCount = await contracts.poolManager.getUserPoolCount(address);
        const poolIds = [];

        for (let i = 0; i < poolCount; i++) {
          const poolId = await contracts.poolManager.getUserPoolAtIndex(address, i);
          poolIds.push(poolId);
        }

        const poolsData = await Promise.all(
          poolIds.map(async (id) => {
            const poolData = await contracts.poolManager.getPool(id);
            return {
              id,
              assets: poolData.assets,
              weights: poolData.weights,
              fee: poolData.fee,
              amplification: poolData.amplification
            };
          })
        );

        setPools(poolsData);
      } catch (error) {
        console.error("Error fetching pools:", error);
      }
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
    connectWallet
  };

  return (
    <Web3Context.Provider value={value}>
      {children}
    </Web3Context.Provider>
  );
}

// Custom hook to use the Web3 context
export function useWeb3() {
  const context = useContext(Web3Context);
  if (!context) {
    throw new Error('useWeb3 must be used within a Web3Provider');
  }
  return context;
}
