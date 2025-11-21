import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { ChakraProvider } from "@chakra-ui/react";
import Dashboard from "../pages/dashboard/Dashboard";

// Mock the web3 hook
jest.mock("../lib/web3-config.jsx", () => ({
  useWeb3: () => ({
    isConnected: true,
    pools: [],
  }),
}));

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
  ResponsiveContainer: ({ children }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  PieChart: ({ children }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  Cell: () => null,
}));

const renderDashboard = () => {
  return render(
    <BrowserRouter>
      <ChakraProvider>
        <Dashboard />
      </ChakraProvider>
    </BrowserRouter>,
  );
};

describe("Dashboard Component", () => {
  it("renders the welcome section", () => {
    renderDashboard();
    expect(screen.getByText("Welcome to Fluxion")).toBeInTheDocument();
    expect(
      screen.getByText(/The next-generation decentralized liquidity protocol/),
    ).toBeInTheDocument();
  });

  it("renders the action buttons", () => {
    renderDashboard();
    expect(screen.getByText("Explore Pools")).toBeInTheDocument();
    expect(screen.getByText("Create Pool")).toBeInTheDocument();
  });

  it("renders the stats overview section", () => {
    renderDashboard();
    expect(screen.getByText("Total Value Locked")).toBeInTheDocument();
    expect(screen.getByText("24h Volume")).toBeInTheDocument();
    expect(screen.getByText("Active Pools")).toBeInTheDocument();
    expect(screen.getByText("Average APY")).toBeInTheDocument();
  });

  it("renders the stats values", () => {
    renderDashboard();
    expect(screen.getByText("$142.5M")).toBeInTheDocument();
    expect(screen.getByText("$28.4M")).toBeInTheDocument();
    expect(screen.getByText("247")).toBeInTheDocument();
    expect(screen.getByText("8.74%")).toBeInTheDocument();
  });

  it("renders the charts", () => {
    renderDashboard();
    expect(screen.getByTestId("line-chart")).toBeInTheDocument();
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
    expect(screen.getByTestId("bar-chart")).toBeInTheDocument();
    expect(screen.getByTestId("pie-chart")).toBeInTheDocument();
  });

  it("renders user positions", () => {
    renderDashboard();
    expect(screen.getByText("ETH-USDC")).toBeInTheDocument();
    expect(screen.getByText("WBTC-ETH")).toBeInTheDocument();
    expect(screen.getByText("$25,000")).toBeInTheDocument();
    expect(screen.getByText("$12,500")).toBeInTheDocument();
  });

  it("renders recent transactions", () => {
    renderDashboard();
    expect(screen.getByText("Add Liquidity")).toBeInTheDocument();
    expect(screen.getByText("Swap")).toBeInTheDocument();
    expect(screen.getByText("Remove Liquidity")).toBeInTheDocument();
  });

  it("handles tab switching", () => {
    renderDashboard();
    const tabs = screen.getAllByRole("tab");

    // Click on the second tab
    fireEvent.click(tabs[1]);

    // Verify the content changes (you might need to adjust this based on your actual tab content)
    expect(screen.getByText("Recent Transactions")).toBeInTheDocument();
  });

  it("renders pool distribution", () => {
    renderDashboard();
    expect(screen.getByText("ETH-USDC")).toBeInTheDocument();
    expect(screen.getByText("WBTC-ETH")).toBeInTheDocument();
    expect(screen.getByText("ETH-DAI")).toBeInTheDocument();
    expect(screen.getByText("USDC-DAI")).toBeInTheDocument();
  });
});
