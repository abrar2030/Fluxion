// web-frontend/src/components/pools/WeightEditor.js
import React, { useState, useEffect } from "react";

const WeightEditor = ({
  weights: initialWeights,
  onChange,
  assetSymbols = [],
}) => {
  const [weights, setWeights] = useState(initialWeights || []);
  const [error, setError] = useState("");

  useEffect(() => {
    // Initialize weights if not provided or length doesn't match assetSymbols
    if (
      assetSymbols.length > 0 &&
      initialWeights.length !== assetSymbols.length
    ) {
      setWeights(
        new Array(assetSymbols.length).fill(100 / assetSymbols.length || 0),
      );
    } else {
      setWeights(initialWeights || []);
    }
  }, [initialWeights, assetSymbols]);

  const handleWeightChange = (index, value) => {
    const newWeights = [...weights];
    const parsedValue = parseFloat(value);
    newWeights[index] = isNaN(parsedValue) ? 0 : parsedValue;
    setWeights(newWeights);
    validateAndPropagate(newWeights);
  };

  const validateAndPropagate = (currentWeights) => {
    const totalWeight = currentWeights.reduce((sum, w) => sum + w, 0);
    if (Math.abs(totalWeight - 100) > 0.01 && currentWeights.length > 0) {
      // Allow for small floating point inaccuracies
      setError(
        `Total weight must be 100%. Current: ${totalWeight.toFixed(2)}%`,
      );
    } else {
      setError("");
    }
    onChange(currentWeights);
  };

  const distributeEqually = () => {
    if (assetSymbols.length > 0) {
      const equalWeight = 100 / assetSymbols.length;
      const newWeights = new Array(assetSymbols.length).fill(equalWeight);
      setWeights(newWeights);
      validateAndPropagate(newWeights);
    }
  };

  if (assetSymbols.length === 0) {
    return <p>Please select assets to assign weights.</p>;
  }

  return (
    <div className="weight-editor">
      <h4>Assign Asset Weights</h4>
      {assetSymbols.map((symbol, index) => (
        <div
          key={index}
          style={{
            marginBottom: "10px",
            display: "flex",
            alignItems: "center",
          }}
        >
          <label
            htmlFor={`weight-${index}`}
            style={{ marginRight: "10px", minWidth: "80px" }}
          >
            {symbol || `Asset ${index + 1}`}:
          </label>
          <input
            type="number"
            id={`weight-${index}`}
            value={weights[index] === undefined ? "" : weights[index]}
            onChange={(e) => handleWeightChange(index, e.target.value)}
            placeholder="e.g., 50"
            min="0"
            max="100"
            step="0.01"
            style={{ padding: "8px", flexGrow: 1 }}
          />
          <span style={{ marginLeft: "5px" }}>%</span>
        </div>
      ))}
      {assetSymbols.length > 0 && (
        <button
          onClick={distributeEqually}
          style={{ marginTop: "10px", padding: "8px 15px" }}
        >
          Distribute Equally
        </button>
      )}
      {error && <p style={{ color: "red", marginTop: "10px" }}>{error}</p>}
      <p style={{ marginTop: "10px" }}>
        Total Weight:{" "}
        {weights.reduce((sum, w) => sum + (parseFloat(w) || 0), 0).toFixed(2)}%
      </p>
    </div>
  );
};

export default WeightEditor;
