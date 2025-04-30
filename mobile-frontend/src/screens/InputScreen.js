import React, { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Snackbar, useTheme } from 'react-native-paper';
import PredictionForm from '../components/PredictionForm';
import LoadingIndicator from '../components/LoadingIndicator';
import { predictEnergy } from '../api/client';

const InputScreen = ({ navigation }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const theme = useTheme(); // Access theme

  const handlePredictionSubmit = async (timestamps, meterIds, contextFeatures) => {
    setIsLoading(true);
    setError(null); // Clear previous errors
    try {
      // TODO: Add actual API endpoint configuration
      // const apiUrl = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000'; // Example
      // const results = await predictEnergy(apiUrl, timestamps, meterIds, contextFeatures);
      
      // Simulate API call for now if no backend is running
      console.log("Simulating API call with:", { timestamps, meterIds, contextFeatures });
      await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate network delay
      const mockResults = {
        predictions: { 'meter1': [10.5, 11.2], 'meter2': [20.1, 21.3], 'meter3': [5.5, 6.0] },
        confidence_intervals: { 'meter1': [[9.8, 11.2], [10.5, 11.9]], 'meter2': [[19.0, 21.2], [20.0, 22.6]], 'meter3': [[5.0, 6.0], [5.5, 6.5]] },
        model_version: 'fluxora-v1.2-lstm'
      };
      // Replace mockResults with actual results when API is integrated
      navigation.navigate('Results', { predictionData: mockResults });

    } catch (err) {
      console.error("API Error:", err);
      setError(err.message || 'Could not fetch prediction. Please check API connection.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      {isLoading ? (
        <LoadingIndicator message="Fetching prediction..." />
      ) : (
        <PredictionForm onSubmit={handlePredictionSubmit} isLoading={isLoading} />
      )}
      <Snackbar
        visible={!!error}
        onDismiss={() => setError(null)}
        action={{
          label: 'Dismiss',
          onPress: () => {
            setError(null);
          },
        }}
        duration={Snackbar.DURATION_LONG} // Or DURATION_MEDIUM / DURATION_SHORT
      >
        {error}
      </Snackbar>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    // Use theme background color dynamically
  },
});

export default InputScreen;

