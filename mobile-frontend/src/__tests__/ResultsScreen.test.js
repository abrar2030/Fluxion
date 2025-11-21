import React from "react";
import { render, fireEvent } from "@testing-library/react-native";
import ResultsScreen from "../screens/ResultsScreen";
import { NavigationContainer } from "@react-navigation/native";
import { Provider as PaperProvider } from "react-native-paper";

// Mock the ResultsDisplay component
jest.mock("../components/ResultsDisplay", () => {
  return function MockResultsDisplay({ results }) {
    return <div testID="results-display">{JSON.stringify(results)}</div>;
  };
});

describe("ResultsScreen", () => {
  const mockNavigation = {
    goBack: jest.fn(),
  };

  const mockRoute = {
    params: {
      predictionData: {
        predictions: { meter1: [10.5, 11.2], meter2: [20.1, 21.3] },
        confidence_intervals: {
          meter1: [
            [9.8, 11.2],
            [10.5, 11.9],
          ],
          meter2: [
            [19.0, 21.2],
            [20.0, 22.6],
          ],
        },
        model_version: "fluxora-v1.2-lstm",
      },
    },
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders the results display with prediction data", () => {
    const { getByTestId } = render(
      <NavigationContainer>
        <PaperProvider>
          <ResultsScreen navigation={mockNavigation} route={mockRoute} />
        </PaperProvider>
      </NavigationContainer>,
    );

    const resultsDisplay = getByTestId("results-display");
    expect(resultsDisplay).toBeTruthy();
    expect(resultsDisplay.props.children).toBe(
      JSON.stringify(mockRoute.params.predictionData),
    );
  });

  it('renders the "Make Another Prediction" button', () => {
    const { getByText } = render(
      <NavigationContainer>
        <PaperProvider>
          <ResultsScreen navigation={mockNavigation} route={mockRoute} />
        </PaperProvider>
      </NavigationContainer>,
    );

    const button = getByText("Make Another Prediction");
    expect(button).toBeTruthy();
  });

  it('navigates back when "Make Another Prediction" button is pressed', () => {
    const { getByText } = render(
      <NavigationContainer>
        <PaperProvider>
          <ResultsScreen navigation={mockNavigation} route={mockRoute} />
        </PaperProvider>
      </NavigationContainer>,
    );

    const button = getByText("Make Another Prediction");
    fireEvent.press(button);

    expect(mockNavigation.goBack).toHaveBeenCalled();
  });

  it("applies theme colors correctly", () => {
    const { getByTestId } = render(
      <NavigationContainer>
        <PaperProvider>
          <ResultsScreen navigation={mockNavigation} route={mockRoute} />
        </PaperProvider>
      </NavigationContainer>,
    );

    const safeArea = getByTestId("safe-area");
    expect(safeArea.props.style).toContainEqual({
      backgroundColor: expect.any(String),
    });
  });
});
