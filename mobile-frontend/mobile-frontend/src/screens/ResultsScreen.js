import React from 'react';
import { View, StyleSheet, Button, SafeAreaView } from 'react-native';
import ResultsDisplay from '../components/ResultsDisplay';

const ResultsScreen = ({ route, navigation }) => {
  const { predictionData } = route.params;

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <ResultsDisplay results={predictionData} />
        <View style={styles.buttonContainer}>
          <Button 
            title="Make Another Prediction" 
            onPress={() => navigation.goBack()} 
            color="#4a90e2"
          />
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f9f9f9',
  },
  container: {
    flex: 1,
  },
  buttonContainer: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#eee',
    backgroundColor: '#fff',
  },
});

export default ResultsScreen;

