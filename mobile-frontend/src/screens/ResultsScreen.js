import React from "react";
import { View, StyleSheet, SafeAreaView } from "react-native";
import { Button, useTheme } from "react-native-paper";
import ResultsDisplay from "../components/ResultsDisplay";

const ResultsScreen = ({ route, navigation }) => {
  const { predictionData } = route.params;
  const theme = useTheme(); // Access theme

  return (
    <SafeAreaView
      style={[styles.safeArea, { backgroundColor: theme.colors.background }]}
    >
      <View style={styles.container}>
        <ResultsDisplay results={predictionData} />
        <View style={styles.buttonContainer}>
          <Button
            mode="contained"
            onPress={() => navigation.goBack()}
            style={styles.button}
            labelStyle={styles.buttonLabel}
          >
            Make Another Prediction
          </Button>
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    // Use theme background color dynamically
  },
  container: {
    flex: 1,
    justifyContent: "space-between", // Push button to the bottom
  },
  buttonContainer: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: "#eee", // Consider using theme.colors.outline or similar
    backgroundColor: "#fff", // Consider using theme.colors.surface or elevation
  },
  button: {
    paddingVertical: 8, // Add vertical padding to button
  },
  buttonLabel: {
    fontSize: 16, // Adjust button text size if needed
  },
});

export default ResultsScreen;
