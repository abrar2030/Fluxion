// web-frontend/src/components/synthetics/SyntheticAssetForm.js
import React, { useState } from "react";

const SyntheticAssetForm = ({ onSubmit }) => {
  const [assetName, setAssetName] = useState("");
  const [oracleAddress, setOracleAddress] = useState("");
  const [collateralRatio, setCollateralRatio] = useState(150); // Default to 150%

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onSubmit) {
      onSubmit({ assetName, oracleAddress, collateralRatio });
    }
    // Reset form
    setAssetName("");
    setOracleAddress("");
    setCollateralRatio(150);
  };

  return (
    <div
      className="synthetic-asset-form"
      style={{
        padding: "20px",
        border: "1px solid #eee",
        borderRadius: "8px",
        marginTop: "20px",
      }}
    >
      <h4>Create New Synthetic Asset</h4>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "15px" }}>
          <label
            htmlFor="synthetic-asset-name"
            style={{ display: "block", marginBottom: "5px" }}
          >
            Asset Name:
          </label>
          <input
            type="text"
            id="synthetic-asset-name"
            value={assetName}
            onChange={(e) => setAssetName(e.target.value)}
            placeholder="e.g., sAAPL, sTSLA"
            required
            style={{ width: "100%", padding: "8px", boxSizing: "border-box" }}
          />
        </div>
        <div style={{ marginBottom: "15px" }}>
          <label
            htmlFor="synthetic-oracle-address"
            style={{ display: "block", marginBottom: "5px" }}
          >
            Oracle Address:
          </label>
          <input
            type="text"
            id="synthetic-oracle-address"
            value={oracleAddress}
            onChange={(e) => setOracleAddress(e.target.value)}
            placeholder="0x..."
            required
            style={{ width: "100%", padding: "8px", boxSizing: "border-box" }}
          />
        </div>
        <div style={{ marginBottom: "15px" }}>
          <label
            htmlFor="synthetic-collateral-ratio"
            style={{ display: "block", marginBottom: "5px" }}
          >
            Collateralization Ratio (%):
          </label>
          <input
            type="number"
            id="synthetic-collateral-ratio"
            value={collateralRatio}
            onChange={(e) => setCollateralRatio(parseFloat(e.target.value))}
            min="100"
            step="1"
            required
            style={{ width: "100%", padding: "8px", boxSizing: "border-box" }}
          />
        </div>
        <button
          type="submit"
          style={{
            padding: "10px 15px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "4px",
          }}
        >
          Create Synthetic Asset
        </button>
      </form>
    </div>
  );
};

export default SyntheticAssetForm;
