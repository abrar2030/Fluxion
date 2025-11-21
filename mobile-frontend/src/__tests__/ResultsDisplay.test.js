import React from "react";
import { render } from "@testing-library/react-native";
import ResultsDisplay from "../components/ResultsDisplay";
import { Provider as PaperProvider } from "react-native-paper";

describe("ResultsDisplay", () => {
  const mockResults = {
    predictions: {
      meter1: [10.5, 11.2],
      meter2: [20.1, 21.3],
    },
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
  };

  it('renders "No prediction results" when results is null', () => {
    const { getByText } = render(
      <PaperProvider>
        <ResultsDisplay results={null} />
      </PaperProvider>,
    );

    expect(getByText("No prediction results available.")).toBeTruthy();
  });

  it("renders the model version", () => {
    const { getByText } = render(
      <PaperProvider>
        <ResultsDisplay results={mockResults} />
      </PaperProvider>,
    );

    expect(getByText("Model Version: fluxora-v1.2-lstm")).toBeTruthy();
  });

  it("renders predictions for each meter", () => {
    const { getByText } = render(
      <PaperProvider>
        <ResultsDisplay results={mockResults} />
      </PaperProvider>,
    );

    // Check meter titles
    expect(getByText("Meter ID: meter1")).toBeTruthy();
    expect(getByText("Meter ID: meter2")).toBeTruthy();

    // Check predictions
    expect(getByText("10.5000")).toBeTruthy();
    expect(getByText("11.2000")).toBeTruthy();
    expect(getByText("20.1000")).toBeTruthy();
    expect(getByText("21.3000")).toBeTruthy();
  });

  it("renders confidence intervals for each prediction", () => {
    const { getByText } = render(
      <PaperProvider>
        <ResultsDisplay results={mockResults} />
      </PaperProvider>,
    );

    // Check confidence intervals
    expect(getByText("95% CI: [9.8000, 11.2000]")).toBeTruthy();
    expect(getByText("95% CI: [10.5000, 11.9000]")).toBeTruthy();
    expect(getByText("95% CI: [19.0000, 21.2000]")).toBeTruthy();
    expect(getByText("95% CI: [20.0000, 22.6000]")).toBeTruthy();
  });

  it("handles array-based predictions structure", () => {
    const arrayResults = {
      predictions: [10.5, 11.2],
      confidence_intervals: [
        [9.8, 11.2],
        [10.5, 11.9],
      ],
      model_version: "fluxora-v1.2-lstm",
    };

    const { getByText } = render(
      <PaperProvider>
        <ResultsDisplay results={arrayResults} />
      </PaperProvider>,
    );

    // Check predictions
    expect(getByText("10.5000")).toBeTruthy();
    expect(getByText("11.2000")).toBeTruthy();

    // Check confidence intervals
    expect(getByText("95% CI: [9.8000, 11.2000]")).toBeTruthy();
    expect(getByText("95% CI: [10.5000, 11.9000]")).toBeTruthy();
  });

  it("applies theme colors correctly", () => {
    const { getByText } = render(
      <PaperProvider>
        <ResultsDisplay results={mockResults} />
      </PaperProvider>,
    );

    const predictionValue = getByText("10.5000");
    expect(predictionValue.props.style).toContainEqual({
      color: expect.any(String),
    });

    const confidenceInterval = getByText("95% CI: [9.8000, 11.2000]");
    expect(confidenceInterval.props.style).toContainEqual({
      color: expect.any(String),
    });
  });
});
