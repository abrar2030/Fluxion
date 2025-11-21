// web-frontend/src/components/dashboard/OverviewChart.js
import React from "react";

// This is a placeholder for a more complex chart.
// You would typically use a charting library like Chart.js, Recharts, or Nivo here.

const OverviewChart = ({ data }) => {
  if (!data || data.length === 0) {
    return <p>No overview data available to display chart.</p>;
  }

  // Example: Simple display of data points
  return (
    <div
      style={{
        border: "1px solid #ccc",
        padding: "20px",
        borderRadius: "8px",
        margin: "20px 0",
      }}
    >
      <h4>Portfolio Overview (Placeholder Chart)</h4>
      <ul>
        {data.map((item, index) => (
          <li key={index}>{`${item.label}: ${item.value}`}</li>
        ))}
      </ul>
      <p style={{ textAlign: "center", color: "#777" }}>
        <em>Chart visualization would appear here.</em>
      </p>
    </div>
  );
};

export default OverviewChart;
