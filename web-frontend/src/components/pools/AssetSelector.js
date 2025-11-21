// web-frontend/src/components/pools/AssetSelector.js
import React, { useState, useEffect } from "react";

// Mock API call for fetching assets - replace with actual API call
const fetchAvailableAssets = async () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([
        {
          id: "ETH",
          name: "Ethereum",
          symbol: "ETH",
          logo: "https://s2.coinmarketcap.com/static/img/coins/64x64/1027.png",
        },
        {
          id: "BTC",
          name: "Bitcoin",
          symbol: "BTC",
          logo: "https://s2.coinmarketcap.com/static/img/coins/64x64/1.png",
        },
        {
          id: "USDC",
          name: "USD Coin",
          symbol: "USDC",
          logo: "https://s2.coinmarketcap.com/static/img/coins/64x64/3408.png",
        },
        {
          id: "DAI",
          name: "Dai",
          symbol: "DAI",
          logo: "https://s2.coinmarketcap.com/static/img/coins/64x64/4943.png",
        },
        {
          id: "LINK",
          name: "Chainlink",
          symbol: "LINK",
          logo: "https://s2.coinmarketcap.com/static/img/coins/64x64/1975.png",
        },
      ]);
    }, 500);
  });
};

const AssetSelector = ({ onChange }) => {
  const [availableAssets, setAvailableAssets] = useState([]);
  const [selectedAssets, setSelectedAssets] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const loadAssets = async () => {
      const assets = await fetchAvailableAssets();
      setAvailableAssets(assets);
    };
    loadAssets();
  }, []);

  const handleSelectAsset = (asset) => {
    if (!selectedAssets.find((a) => a.id === asset.id)) {
      const newSelectedAssets = [...selectedAssets, asset];
      setSelectedAssets(newSelectedAssets);
      onChange(newSelectedAssets.map((a) => a.id)); // Pass asset IDs
    }
  };

  const handleRemoveAsset = (assetId) => {
    const newSelectedAssets = selectedAssets.filter((a) => a.id !== assetId);
    setSelectedAssets(newSelectedAssets);
    onChange(newSelectedAssets.map((a) => a.id)); // Pass asset IDs
  };

  const filteredAssets = availableAssets.filter(
    (asset) =>
      asset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      asset.symbol.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  return (
    <div className="asset-selector">
      <h4>Select Assets for Pool</h4>
      <input
        type="text"
        placeholder="Search assets..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        style={{
          marginBottom: "10px",
          padding: "8px",
          width: "calc(100% - 16px)",
        }}
      />
      <div className="selected-assets-display" style={{ marginBottom: "10px" }}>
        {selectedAssets.length > 0 ? (
          selectedAssets.map((asset) => (
            <span
              key={asset.id}
              className="selected-asset-tag"
              style={{
                display: "inline-flex",
                alignItems: "center",
                padding: "5px 10px",
                margin: "5px",
                backgroundColor: "#e0e0e0",
                borderRadius: "15px",
                fontSize: "0.9rem",
              }}
            >
              <img
                src={asset.logo}
                alt={asset.name}
                style={{ width: "20px", height: "20px", marginRight: "8px" }}
              />
              {asset.symbol}
              <button
                onClick={() => handleRemoveAsset(asset.id)}
                style={{
                  marginLeft: "10px",
                  border: "none",
                  backgroundColor: "transparent",
                  color: "#ff0000",
                  cursor: "pointer",
                  fontSize: "1rem",
                }}
              >
                &times;
              </button>
            </span>
          ))
        ) : (
          <p>No assets selected.</p>
        )}
      </div>
      <div
        className="available-assets-list"
        style={{
          maxHeight: "200px",
          overflowY: "auto",
          border: "1px solid #ccc",
          padding: "10px",
        }}
      >
        {filteredAssets.map((asset) => (
          <div
            key={asset.id}
            onClick={() => handleSelectAsset(asset)}
            style={{
              padding: "10px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              borderBottom: "1px solid #eee",
            }}
            className="asset-item"
          >
            <img
              src={asset.logo}
              alt={asset.name}
              style={{ width: "24px", height: "24px", marginRight: "10px" }}
            />
            <span>
              {asset.name} ({asset.symbol})
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AssetSelector;
