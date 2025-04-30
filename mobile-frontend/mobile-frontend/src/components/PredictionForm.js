import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, ScrollView, TouchableOpacity, Alert } from 'react-native';

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [timestamps, setTimestamps] = useState('');
  const [meterIds, setMeterIds] = useState('');
  const [contextFeatures, setContextFeatures] = useState('');

  const handleSubmit = () => {
    try {
      // Validate inputs
      if (!timestamps.trim() || !meterIds.trim() || !contextFeatures.trim()) {
        Alert.alert('Validation Error', 'All fields are required');
        return;
      }

      // Parse inputs
      const timestampArray = timestamps.split(',').map(t => t.trim());
      const meterIdArray = meterIds.split(',').map(id => id.trim());
      let contextFeaturesObj;
      
      try {
        contextFeaturesObj = JSON.parse(contextFeatures);
      } catch (e) {
        Alert.alert('Invalid Format', 'Context features must be valid JSON');
        return;
      }

      // Submit data
      onSubmit(timestampArray, meterIdArray, contextFeaturesObj);
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.formGroup}>
        <Text style={styles.label}>Timestamps (comma-separated)</Text>
        <TextInput
          style={styles.input}
          value={timestamps}
          onChangeText={setTimestamps}
          placeholder="2023-01-01T12:00:00, 2023-01-01T13:00:00"
          multiline
        />
      </View>

      <View style={styles.formGroup}>
        <Text style={styles.label}>Meter IDs (comma-separated)</Text>
        <TextInput
          style={styles.input}
          value={meterIds}
          onChangeText={setMeterIds}
          placeholder="meter1, meter2, meter3"
          multiline
        />
      </View>

      <View style={styles.formGroup}>
        <Text style={styles.label}>Context Features (JSON format)</Text>
        <TextInput
          style={[styles.input, styles.jsonInput]}
          value={contextFeatures}
          onChangeText={setContextFeatures}
          placeholder='{"temperature": 22.5, "humidity": 65, "occupancy": 4}'
          multiline
          numberOfLines={4}
        />
      </View>

      <TouchableOpacity 
        style={[styles.button, isLoading && styles.buttonDisabled]} 
        onPress={handleSubmit}
        disabled={isLoading}
      >
        <Text style={styles.buttonText}>
          {isLoading ? 'Getting Prediction...' : 'Get Prediction'}
        </Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  formGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#fff',
  },
  jsonInput: {
    height: 120,
    textAlignVertical: 'top',
  },
  button: {
    backgroundColor: '#4a90e2',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 10,
  },
  buttonDisabled: {
    backgroundColor: '#a0c4e7',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default PredictionForm;
