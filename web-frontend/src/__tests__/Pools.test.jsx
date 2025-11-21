import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { ChakraProvider } from "@chakra-ui/react";
import Pools from "../pages/pools/Pools";

// Mock the web3 hook
jest.mock("../lib/web3-config.jsx", () => ({
  useWeb3: () => ({
    isConnected: true,
    pools: [],
  }),
}));

// Mock the recharts components
jest.mock("recharts", () => ({
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
}));

const renderPools = () => {
  return render(
    <BrowserRouter>
      <ChakraProvider>
        <Pools />
      </ChakraProvider>
    </BrowserRouter>,
  );
};

describe("Pools Component", () => {
  it("renders the hero section", () => {
    renderPools();
    expect(screen.getByText("Liquidity Pools")).toBeInTheDocument();
    expect(
      screen.getByText(/Provide liquidity to earn fees/),
    ).toBeInTheDocument();
  });

  it("renders the create pool button", () => {
    renderPools();
    expect(screen.getByText("Create Pool")).toBeInTheDocument();
  });

  it("renders the search input", () => {
    renderPools();
    const searchInput = screen.getByPlaceholderText("Search pools...");
    expect(searchInput).toBeInTheDocument();
  });

  it("filters pools by search query", () => {
    renderPools();
    const searchInput = screen.getByPlaceholderText("Search pools...");

    // Search for ETH-USDC pool
    fireEvent.change(searchInput, { target: { value: "ETH-USDC" } });
    expect(screen.getByText("ETH-USDC")).toBeInTheDocument();
    expect(screen.queryByText("WBTC-ETH")).not.toBeInTheDocument();
  });

  it("filters pools by type", () => {
    renderPools();
    const filterButton = screen.getByText("All Pools");
    fireEvent.click(filterButton);

    // Select weighted pools
    const weightedOption = screen.getByText("Weighted Pools");
    fireEvent.click(weightedOption);

    // Check if only weighted pools are shown
    expect(screen.getByText("ETH-USDC")).toBeInTheDocument();
    expect(screen.getByText("WBTC-ETH")).toBeInTheDocument();
    expect(screen.queryByText("USDC-DAI")).not.toBeInTheDocument();
  });

  it("sorts pools by TVL", () => {
    renderPools();
    const tvlHeader = screen.getByText("TVL");
    fireEvent.click(tvlHeader);

    // Get all TVL values
    const tvlValues = screen.getAllByText(/\$\d+\.\d+M/);

    // Check if values are in descending order
    const values = tvlValues.map((el) =>
      parseFloat(el.textContent.replace("$", "").replace("M", "")),
    );
    expect(values).toEqual([...values].sort((a, b) => b - a));
  });

  it("sorts pools by APY", () => {
    renderPools();
    const apyHeader = screen.getByText("APY");
    fireEvent.click(apyHeader);

    // Get all APY values
    const apyValues = screen.getAllByText(/\d+\.\d+%/);

    // Check if values are in descending order
    const values = apyValues.map((el) =>
      parseFloat(el.textContent.replace("%", "")),
    );
    expect(values).toEqual([...values].sort((a, b) => b - a));
  });

  it("opens pool details modal on pool click", () => {
    renderPools();
    const poolRow = screen.getByText("ETH-USDC").closest("tr");
    fireEvent.click(poolRow);

    // Check if modal is opened with pool details
    expect(screen.getByText("Pool Details")).toBeInTheDocument();
    expect(screen.getByText("ETH-USDC")).toBeInTheDocument();
  });

  it("renders pool statistics", () => {
    renderPools();
    expect(screen.getByText("$4.2M")).toBeInTheDocument();
    expect(screen.getByText("$1.2M")).toBeInTheDocument();
    expect(screen.getByText("8.4%")).toBeInTheDocument();
  });

  it("renders pool type badges", () => {
    renderPools();
    expect(screen.getByText("Weighted")).toBeInTheDocument();
    expect(screen.getByText("Stable")).toBeInTheDocument();
  });

  it("renders user liquidity information", () => {
    renderPools();
    expect(screen.getByText("$25,000")).toBeInTheDocument();
    expect(screen.getByText("$12,500")).toBeInTheDocument();
  });
});
