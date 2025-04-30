import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';

const ResultsDisplay = ({ results }) => {
  if (!results) {
    return null; // Don't render anything if there are no results
  }

  const { predictions, confidence_intervals, model_version } = results;

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Prediction Results</Text>
      <Text style={styles.modelVersion}>Model Version: {model_version}</Text>

      {predictions.map((prediction, index) => (
        <View key={index} style={styles.resultItem}>
          <Text style={styles.predictionLabel}>Prediction {index + 1}:</Text>
          <Text style={styles.predictionValue}>{prediction.toFixed(4)}</Text>
          {confidence_intervals && confidence_intervals[index] && (
            <Text style={styles.confidenceInterval}>
              95% CI: [{confidence_intervals[index][0].toFixed(4)}, {confidence_intervals[index][1].toFixed(4)}]
            </Text>
          )}
        </View>
      ))}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f9f9f9',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
    textAlign: 'center',
  },
  modelVersion: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  resultItem: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#eee',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  predictionLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
    color: '#555',
  },
  predictionValue: {
    fontSize: 18,
    color: '#4a90e2',
    marginBottom: 4,
  },
  confidenceInterval: {
    fontSize: 14,
    color: '#777',
  },
});

export default ResultsDisplay;

