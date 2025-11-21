import React from "react";
import { render, fireEvent, waitFor } from "@testing-library/react-native";
import InputScreen from "../screens/InputScreen";
import { NavigationContainer } from "@react-navigation/native";
import { Provider as PaperProvider } from "react-native-paper";

// Mock the navigation
const mockNavigation = {
  navigate: jest.fn(),
};

// Mock the PredictionForm component
jest.mock("../components/PredictionForm", () => {
  return function MockPredictionForm({ onSubmit, isLoading }) {
    return (
      <button
        testID="submit-button"
        onPress={() =>
          onSubmit(["2024-01-01", "2024-01-02"], ["meter1", "meter2"], {
            temperature: 25,
            humidity: 60,
          })
        }
        disabled={isLoading}
      >
        Submit
      </button>
    );
  };
});

// Mock the LoadingIndicator component
jest.mock("../components/LoadingIndicator", () => {
  return function MockLoadingIndicator({ message }) {
    return <div testID="loading-indicator">{message}</div>;
  };
});

describe("InputScreen", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders the prediction form", () => {
    const { getByTestId } = render(
      <NavigationContainer>
        <PaperProvider>
          <InputScreen navigation={mockNavigation} />
        </PaperProvider>
      </NavigationContainer>,
    );

    expect(getByTestId("submit-button")).toBeTruthy();
  });

  it("shows loading indicator when submitting", async () => {
    const { getByTestId, queryByTestId } = render(
      <NavigationContainer>
        <PaperProvider>
          <InputScreen navigation={mockNavigation} />
        </PaperProvider>
      </NavigationContainer>,
    );

    fireEvent.press(getByTestId("submit-button"));

    await waitFor(() => {
      expect(queryByTestId("loading-indicator")).toBeTruthy();
    });
  });

  it("navigates to Results screen after successful submission", async () => {
    const { getByTestId } = render(
      <NavigationContainer>
        <PaperProvider>
          <InputScreen navigation={mockNavigation} />
        </PaperProvider>
      </NavigationContainer>,
    );

    fireEvent.press(getByTestId("submit-button"));

    await waitFor(() => {
      expect(mockNavigation.navigate).toHaveBeenCalledWith(
        "Results",
        expect.any(Object),
      );
    });
  });

  it("shows error message when submission fails", async () => {
    // Mock console.error to prevent error output in tests
    const originalConsoleError = console.error;
    console.error = jest.fn();

    const { getByTestId, getByText } = render(
      <NavigationContainer>
        <PaperProvider>
          <InputScreen navigation={mockNavigation} />
        </PaperProvider>
      </NavigationContainer>,
    );

    // Simulate an error by making the navigation throw
    mockNavigation.navigate.mockImplementationOnce(() => {
      throw new Error("API Error");
    });

    fireEvent.press(getByTestId("submit-button"));

    await waitFor(() => {
      expect(
        getByText("Could not fetch prediction. Please check API connection."),
      ).toBeTruthy();
    });

    // Restore console.error
    console.error = originalConsoleError;
  });
});
