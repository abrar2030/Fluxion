import React, { useState } from "react";
import { View, StyleSheet } from "react-native";
import { Snackbar, useTheme } from "react-native-paper";
import PredictionForm from "../components/PredictionForm";
import LoadingIndicator from "../components/LoadingIndicator";
import { predictEnergy } from "../api/client";

const InputScreen = ({ navigation }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const theme = useTheme();

  const handlePredictionSubmit = async (
    timestamps,
    meterIds,
    contextFeatures,
  ) => {
    setIsLoading(true);
    setError(null); // Clear previous errors

    try {
      // Use the predictEnergy function from our API client
      const results = await predictEnergy(
        timestamps,
        meterIds,
        contextFeatures,
      );

      // Navigate to results screen with the prediction data
      navigation.navigate("Results", { predictionData: results });
    } catch (err) {
      console.error("API Error:", err);
      setError(
        err.message ||
          "Could not fetch prediction. Please check API connection.",
      );

      // If in development mode, provide fallback mock data for testing
      if (__DEV__) {
        console.log("Development mode: Using mock data as fallback");
        const mockResults = {
          predictions: {
            meter1: [10.5, 11.2],
            meter2: [20.1, 21.3],
            meter3: [5.5, 6.0],
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
            meter3: [
              [5.0, 6.0],
              [5.5, 6.5],
            ],
          },
          model_version: "fluxora-v1.2-lstm",
        };
        navigation.navigate("Results", { predictionData: mockResults });
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View
      style={[styles.container, { backgroundColor: theme.colors.background }]}
    >
      {isLoading ? (
        <LoadingIndicator message="Fetching prediction..." />
      ) : (
        <PredictionForm
          onSubmit={handlePredictionSubmit}
          isLoading={isLoading}
        />
      )}
      <Snackbar
        visible={!!error}
        onDismiss={() => setError(null)}
        action={{
          label: "Dismiss",
          onPress: () => {
            setError(null);
          },
        }}
        duration={Snackbar.DURATION_LONG}
      >
        {error}
      </Snackbar>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
});

export default InputScreen;
