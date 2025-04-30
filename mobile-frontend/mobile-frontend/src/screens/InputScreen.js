import React, { useState } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import PredictionForm from '../components/PredictionForm';
import LoadingIndicator from '../components/LoadingIndicator';
import { predictEnergy } from '../api/client';

const InputScreen = ({ navigation }) => {
  const [isLoading, setIsLoading] = useState(false);

  const handlePredictionSubmit = async (timestamps, meterIds, contextFeatures) => {
    setIsLoading(true);
    try {
      const results = await predictEnergy(timestamps, meterIds, contextFeatures);
      navigation.navigate('Results', { predictionData: results });
    } catch (error) {
      Alert.alert('API Error', error.message || 'Could not fetch prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {isLoading ? (
        <LoadingIndicator message="Fetching prediction..." />
      ) : (
        <PredictionForm onSubmit={handlePredictionSubmit} isLoading={isLoading} />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f0f2f5', // Light background for the screen
  },
});

export default InputScreen;

