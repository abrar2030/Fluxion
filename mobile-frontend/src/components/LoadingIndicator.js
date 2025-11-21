import React from "react";
import { View, StyleSheet } from "react-native";
import { ActivityIndicator, Text, useTheme } from "react-native-paper";

const LoadingIndicator = ({ message = "Loading..." }) => {
  const theme = useTheme(); // Access theme colors

  return (
    <View style={styles.container}>
      <ActivityIndicator
        animating={true}
        size="large"
        color={theme.colors.primary}
      />
      <Text style={[styles.message, { color: theme.colors.onSurfaceVariant }]}>
        {message}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1, // Make it take full space if used directly in a screen
    padding: 20,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "transparent", // Or use theme.colors.background if needed
  },
  message: {
    marginTop: 15,
    fontSize: 16,
  },
});

export default LoadingIndicator;
