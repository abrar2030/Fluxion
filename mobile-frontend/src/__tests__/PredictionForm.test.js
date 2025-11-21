import React from "react";
import { render, fireEvent, waitFor } from "@testing-library/react-native";
import PredictionForm from "../components/PredictionForm";
import { Provider as PaperProvider } from "react-native-paper";

// Mock the date-fns format function
jest.mock("date-fns", () => ({
  format: (date) => date.toISOString(),
}));

describe("PredictionForm", () => {
  const mockOnSubmit = jest.fn();
  const defaultProps = {
    onSubmit: mockOnSubmit,
    isLoading: false,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders all form elements", () => {
    const { getByText, getByPlaceholderText } = render(
      <PaperProvider>
        <PredictionForm {...defaultProps} />
      </PaperProvider>,
    );

    expect(getByText("Timestamps")).toBeTruthy();
    expect(getByText("Add Timestamp")).toBeTruthy();
    expect(getByPlaceholderText("meter1, meter2, meter3")).toBeTruthy();
    expect(
      getByPlaceholderText('{"temperature": 22.5, "humidity": 65}'),
    ).toBeTruthy();
    expect(getByText("Get Prediction")).toBeTruthy();
  });

  it("shows validation errors for empty fields", () => {
    const { getByText } = render(
      <PaperProvider>
        <PredictionForm {...defaultProps} />
      </PaperProvider>,
    );

    fireEvent.press(getByText("Get Prediction"));

    expect(getByText("At least one timestamp is required.")).toBeTruthy();
    expect(getByText("Meter IDs are required.")).toBeTruthy();
    expect(getByText("Context Features are required.")).toBeTruthy();
  });

  it("validates JSON format for context features", () => {
    const { getByText, getByPlaceholderText } = render(
      <PaperProvider>
        <PredictionForm {...defaultProps} />
      </PaperProvider>,
    );

    const contextInput = getByPlaceholderText(
      '{"temperature": 22.5, "humidity": 65}',
    );
    fireEvent.changeText(contextInput, "invalid json");
    fireEvent.press(getByText("Get Prediction"));

    expect(getByText("Context Features must be valid JSON.")).toBeTruthy();
  });

  it("submits form with valid data", async () => {
    const { getByText, getByPlaceholderText } = render(
      <PaperProvider>
        <PredictionForm {...defaultProps} />
      </PaperProvider>,
    );

    // Add a timestamp
    const addButton = getByText("Add Timestamp");
    fireEvent.press(addButton);

    // Mock the date picker confirmation
    const mockDate = new Date("2024-01-01T12:00:00");
    const datePicker = getByText("Add Timestamp");
    fireEvent(datePicker, "onConfirm", mockDate);

    // Fill in meter IDs
    const meterInput = getByPlaceholderText("meter1, meter2, meter3");
    fireEvent.changeText(meterInput, "meter1, meter2");

    // Fill in context features
    const contextInput = getByPlaceholderText(
      '{"temperature": 22.5, "humidity": 65}',
    );
    fireEvent.changeText(contextInput, '{"temperature": 25, "humidity": 60}');

    // Submit the form
    fireEvent.press(getByText("Get Prediction"));

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        [mockDate.toISOString()],
        ["meter1", "meter2"],
        { temperature: 25, humidity: 60 },
      );
    });
  });

  it("disables submit button when loading", () => {
    const { getByText } = render(
      <PaperProvider>
        <PredictionForm {...defaultProps} isLoading={true} />
      </PaperProvider>,
    );

    const submitButton = getByText("Getting Prediction...");
    expect(submitButton.props.disabled).toBe(true);
  });

  it("removes timestamp when clicking close on chip", () => {
    const { getByText, queryByText } = render(
      <PaperProvider>
        <PredictionForm {...defaultProps} />
      </PaperProvider>,
    );

    // Add a timestamp
    const addButton = getByText("Add Timestamp");
    fireEvent.press(addButton);

    // Mock the date picker confirmation
    const mockDate = new Date("2024-01-01T12:00:00");
    const datePicker = getByText("Add Timestamp");
    fireEvent(datePicker, "onConfirm", mockDate);

    // Find and click the close button on the chip
    const chip = getByText(mockDate.toISOString());
    const closeButton = chip.parent.props.onClose;
    fireEvent(closeButton);

    // Verify the timestamp was removed
    expect(queryByText(mockDate.toISOString())).toBeNull();
  });
});
