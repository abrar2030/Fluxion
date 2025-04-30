import React, { useState, useEffect } from 'react';
import { useWeb3 } from .../lib/web3-config.jsx';
import HeatmapChart from './HeatmapChart';

export default function RiskMatrix() {
  const { pools } = useWeb3();
  const [riskData, setRiskData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('/api/risk-metrics');
      const data = await response.json();
      setRiskData(data.map(pool => ({
        id: pool.id,
        var: pool.var_95,
        cvar: pool.cvar,
        liquidityRisk: pool.liquidity_at_risk
      })));
    };
    fetchData();
  }, [pools]);

  return (
    <div className="risk-matrix-container">
      <h2>Risk Analysis Matrix</h2>
      <HeatmapChart data={riskData} x="id" y="metric" value="value" />
    </div>
  );
}
