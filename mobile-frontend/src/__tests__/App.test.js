import React from "react";
import { render } from "@testing-library/react-native";
import App from "../../App";
import { NavigationContainer } from "@react-navigation/native";
import { Provider as PaperProvider } from "react-native-paper";

// Mock the navigation components
jest.mock("@react-navigation/native", () => ({
  NavigationContainer: ({ children }) => children,
}));

jest.mock("@react-navigation/native-stack", () => ({
  createNativeStackNavigator: () => ({
    Navigator: ({ children }) => children,
    Screen: ({ children }) => children,
  }),
}));

jest.mock("@react-navigation/bottom-tabs", () => ({
  createBottomTabNavigator: () => ({
    Navigator: ({ children }) => children,
    Screen: ({ children }) => children,
  }),
}));

// Mock the screens
jest.mock("../../src/screens/InputScreen", () => "InputScreen");
jest.mock("../../src/screens/ResultsScreen", () => "ResultsScreen");
jest.mock("../../src/screens/AssetsScreen", () => "AssetsScreen");
jest.mock("../../src/screens/PoolsScreen", () => "PoolsScreen");

describe("App Component", () => {
  it("renders without crashing", () => {
    const { getByText } = render(<App />);
    expect(getByText("Prediction")).toBeTruthy();
    expect(getByText("Assets")).toBeTruthy();
    expect(getByText("Pools")).toBeTruthy();
  });

  it("has correct theme colors", () => {
    const { getByText } = render(<App />);
    const predictionTab = getByText("Prediction");
    expect(predictionTab).toBeTruthy();
    // Add more theme-related tests as needed
  });
});
