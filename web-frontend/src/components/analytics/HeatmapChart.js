// web-frontend/src/components/analytics/HeatmapChart.js
import React from "react";
import { HeatMapGrid } from "react-grid-heatmap";

// Helper function to transform data for the heatmap
const transformDataForHeatmap = (data, xKey, yKey, valueKey) => {
  if (!data || data.length === 0) {
    return {
      data: [],
      xLabels: [],
      yLabels: [],
    };
  }

  const xLabels = Array.from(new Set(data.map((item) => item[xKey])));
  const yLabels = Array.from(new Set(data.map((item) => item[yKey])));

  const heatmapData = new Array(yLabels.length)
    .fill(0)
    .map(() => new Array(xLabels.length).fill(0));

  data.forEach((item) => {
    const xIndex = xLabels.indexOf(item[xKey]);
    const yIndex = yLabels.indexOf(item[yKey]);
    if (xIndex !== -1 && yIndex !== -1) {
      heatmapData[yIndex][xIndex] = item[valueKey] || 0;
    }
  });

  return {
    data: heatmapData,
    xLabels,
    yLabels,
  };
};

const HeatmapChart = ({ data, x = "x", y = "y", value = "value" }) => {
  if (!data || data.length === 0) {
    return <p>No data available for heatmap.</p>;
  }

  const {
    data: heatmapData,
    xLabels,
    yLabels,
  } = transformDataForHeatmap(data, x, y, value);

  if (
    heatmapData.length === 0 ||
    xLabels.length === 0 ||
    yLabels.length === 0
  ) {
    return <p>Insufficient data to render heatmap.</p>;
  }

  return (
    <div style={{ width: "100%", fontFamily: "sans-serif" }}>
      <HeatMapGrid
        data={heatmapData}
        xLabels={xLabels}
        yLabels={yLabels}
        // Style options (can be customized further)
        cellStyle={(_x, _y, ratio) => ({
          background: `rgb(12, 160, 44, ${ratio})`,
          fontSize: "0.8rem",
          color: `rgb(0, 0, 0, ${ratio / 2 + 0.4})`,
        })}
        cellHeight="2.5rem"
        xLabelsStyle={(index) => ({
          color: index % 2 ? "transparent" : "#777",
          fontSize: ".8rem",
        })}
        yLabelsStyle={() => ({
          fontSize: ".7rem",
          textTransform: "uppercase",
          color: "#777",
        })}
        square // Makes cells square
      />
    </div>
  );
};

export default HeatmapChart;
