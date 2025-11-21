import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { ChakraProvider } from "@chakra-ui/react";
import Synthetics from "../pages/synthetics/Synthetics";

// Mock the web3 hook
jest.mock("../lib/web3-config.jsx", () => ({
  useWeb3: () => ({
    isConnected: true,
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

const renderSynthetics = () => {
  return render(
    <BrowserRouter>
      <ChakraProvider>
        <Synthetics />
      </ChakraProvider>
    </BrowserRouter>,
  );
};

describe("Synthetics Component", () => {
  it("renders the hero section", () => {
    renderSynthetics();
    expect(screen.getByText("Synthetic Assets")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Trade synthetic assets that track the price of real-world assets/,
      ),
    ).toBeInTheDocument();
  });

  it("renders the mint synthetic button", () => {
    renderSynthetics();
    expect(screen.getByText("Mint Synthetic")).toBeInTheDocument();
  });

  it("renders synthetic assets list", () => {
    renderSynthetics();
    expect(screen.getByText("sETH")).toBeInTheDocument();
    expect(screen.getByText("sBTC")).toBeInTheDocument();
    expect(screen.getByText("sGOLD")).toBeInTheDocument();
    expect(screen.getByText("sEUR")).toBeInTheDocument();
    expect(screen.getByText("sTSLA")).toBeInTheDocument();
  });

  it("displays synthetic asset details", () => {
    renderSynthetics();
    expect(screen.getByText("$1,720.45")).toBeInTheDocument();
    expect(screen.getByText("$42,350.78")).toBeInTheDocument();
    expect(screen.getByText("$1,845.20")).toBeInTheDocument();
  });

  it("shows price changes with correct indicators", () => {
    renderSynthetics();
    expect(screen.getByText("+2.4%")).toBeInTheDocument();
    expect(screen.getByText("-0.5%")).toBeInTheDocument();
  });

  it("displays TVL and volume information", () => {
    renderSynthetics();
    expect(screen.getByText("$42.5M")).toBeInTheDocument();
    expect(screen.getByText("$8.2M")).toBeInTheDocument();
    expect(screen.getByText("$68.3M")).toBeInTheDocument();
    expect(screen.getByText("$12.5M")).toBeInTheDocument();
  });

  it("shows collateralization ratios", () => {
    renderSynthetics();
    expect(screen.getByText("150%")).toBeInTheDocument();
    expect(screen.getByText("175%")).toBeInTheDocument();
    expect(screen.getByText("120%")).toBeInTheDocument();
  });

  it("renders user positions", () => {
    renderSynthetics();
    expect(screen.getByText("5.2")).toBeInTheDocument();
    expect(screen.getByText("$8,946.34")).toBeInTheDocument();
    expect(screen.getByText("$13,419.51")).toBeInTheDocument();
  });

  it("opens synthetic details modal on click", () => {
    renderSynthetics();
    const syntheticRow = screen.getByText("sETH").closest("tr");
    fireEvent.click(syntheticRow);

    expect(screen.getByText("Synthetic Details")).toBeInTheDocument();
    expect(
      screen.getByText(/Synthetic Ethereum that tracks the price of ETH/),
    ).toBeInTheDocument();
  });

  it("renders price and volume charts", () => {
    renderSynthetics();
    expect(screen.getByTestId("line-chart")).toBeInTheDocument();
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
  });
});
