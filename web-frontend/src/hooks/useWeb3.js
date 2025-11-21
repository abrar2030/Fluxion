import { useState, useEffect } from "react";
import { ethers } from "ethers";

export const useWeb3 = () => {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [account, setAccount] = useState(null);
  const [chainId, setChainId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initWeb3 = async () => {
      try {
        if (window.ethereum) {
          const provider = new ethers.BrowserProvider(window.ethereum);
          setProvider(provider);

          // Request account access
          const accounts = await provider.send("eth_requestAccounts", []);
          setAccount(accounts[0]);

          // Get signer
          const signer = await provider.getSigner();
          setSigner(signer);

          // Get chain ID
          const network = await provider.getNetwork();
          setChainId(network.chainId);

          // Listen for account changes
          window.ethereum.on("accountsChanged", (accounts) => {
            setAccount(accounts[0]);
          });

          // Listen for chain changes
          window.ethereum.on("chainChanged", (chainId) => {
            setChainId(chainId);
            window.location.reload();
          });
        } else {
          setError(new Error("Please install MetaMask"));
        }
      } catch (err) {
        setError(err);
      } finally {
        setIsLoading(false);
      }
    };

    initWeb3();

    return () => {
      if (window.ethereum) {
        window.ethereum.removeAllListeners();
      }
    };
  }, []);

  return {
    provider,
    signer,
    account,
    chainId,
    isLoading,
    error,
  };
};
