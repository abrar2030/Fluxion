import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { ChakraProvider } from "@chakra-ui/react";
import Analytics from "../pages/analytics/Analytics";

// Mock the recharts components
jest.mock("recharts", () => ({
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  AreaChart: ({ children }) => <div data-testid="area-chart">{children}</div>,
  Area: () => null,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  PieChart: ({ children }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  Cell: () => null,
}));

const renderAnalytics = () => {
  return render(
    <BrowserRouter>
      <ChakraProvider>
        <Analytics />
      </ChakraProvider>
    </BrowserRouter>,
  );
};

describe("Analytics Component", () => {
  it("renders the hero section", () => {
    renderAnalytics();
    expect(screen.getByText("Analytics Dashboard")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Comprehensive analytics and insights for the Fluxion ecosystem/,
      ),
    ).toBeInTheDocument();
  });

  it("renders the stats overview section", () => {
    renderAnalytics();
    expect(screen.getByText("Total Value Locked")).toBeInTheDocument();
    expect(screen.getByText("24h Volume")).toBeInTheDocument();
    expect(screen.getByText("Active Pools")).toBeInTheDocument();
    expect(screen.getByText("Average APY")).toBeInTheDocument();
  });

  it("displays correct stat values", () => {
    renderAnalytics();
    expect(screen.getByText("$142.5M")).toBeInTheDocument();
    expect(screen.getByText("$28.4M")).toBeInTheDocument();
    expect(screen.getByText("247")).toBeInTheDocument();
  });

  it("shows stat trends with arrows", () => {
    renderAnalytics();
    const increaseArrows = screen.getAllByText("23.36%");
    const decreaseArrows = screen.getAllByText("5.14%");
    expect(increaseArrows.length).toBeGreaterThan(0);
    expect(decreaseArrows.length).toBeGreaterThan(0);
  });

  it("renders top pools table", () => {
    renderAnalytics();
    expect(screen.getByText("ETH-USDC")).toBeInTheDocument();
    expect(screen.getByText("WBTC-ETH")).toBeInTheDocument();
    expect(screen.getByText("USDC-DAI-USDT")).toBeInTheDocument();
    expect(screen.getByText("ETH-LINK")).toBeInTheDocument();
  });

  it("displays pool statistics", () => {
    renderAnalytics();
    expect(screen.getByText("$42.5M")).toBeInTheDocument();
    expect(screen.getByText("$8.2M")).toBeInTheDocument();
    expect(screen.getByText("8.4%")).toBeInTheDocument();
  });

  it("renders pool type distribution", () => {
    renderAnalytics();
    expect(screen.getByText("Weighted")).toBeInTheDocument();
    expect(screen.getByText("Stable")).toBeInTheDocument();
    expect(screen.getByText("Boosted")).toBeInTheDocument();
  });

  it("renders all chart types", () => {
    renderAnalytics();
    expect(screen.getByTestId("line-chart")).toBeInTheDocument();
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
    expect(screen.getByTestId("bar-chart")).toBeInTheDocument();
    expect(screen.getByTestId("pie-chart")).toBeInTheDocument();
  });

  it("expands chart on click", () => {
    renderAnalytics();
    const chart = screen.getByTestId("line-chart");
    fireEvent.click(chart);

    // Check if modal is opened
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("displays pool trends", () => {
    renderAnalytics();
    const trendUp = screen.getAllByText("up");
    const trendDown = screen.getAllByText("down");
    expect(trendUp.length).toBeGreaterThan(0);
    expect(trendDown.length).toBeGreaterThan(0);
  });
});
