import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import {
  Provider as PaperProvider,
  DefaultTheme,
  useTheme,
} from "react-native-paper";
import MaterialCommunityIcons from "react-native-vector-icons/MaterialCommunityIcons";

// Import Screens
import InputScreen from "./src/screens/InputScreen";
import ResultsScreen from "./src/screens/ResultsScreen";
import AssetsScreen from "./src/screens/AssetsScreen";
import PoolsScreen from "./src/screens/PoolsScreen";

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

// Define a custom theme
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: "#6200ee",
    accent: "#03dac4",
    background: "#f6f6f6",
    secondaryContainer: "#e0d6ff", // Example for active tab background
    onSecondaryContainer: "#21005d", // Example for active tab icon/label
    onSurfaceVariant: "#49454f", // Example for inactive tab icon/label
  },
};

// Stack Navigator for the Prediction Flow
function PredictionStack() {
  const { colors } = useTheme(); // Use theme from PaperProvider
  return (
    <Stack.Navigator
      initialRouteName="Input"
      screenOptions={{
        headerStyle: {
          backgroundColor: colors.primary,
        },
        headerTintColor: "#fff",
        headerTitleStyle: {
          fontWeight: "bold",
        },
      }}
    >
      <Stack.Screen
        name="Input"
        component={InputScreen}
        options={{ title: "Energy Prediction Input" }}
      />
      <Stack.Screen
        name="Results"
        component={ResultsScreen}
        options={{ title: "Prediction Results" }}
      />
    </Stack.Navigator>
  );
}

// Main App component with Bottom Tab Navigator
export default function App() {
  return (
    <PaperProvider theme={theme}>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            headerShown: false, // Hide header for tabs, PredictionStack has its own
            tabBarIcon: ({ focused, color, size }) => {
              let iconName;
              if (route.name === "Predict") {
                iconName = focused ? "chart-line" : "chart-line-variant";
              } else if (route.name === "Assets") {
                iconName = focused ? "bitcoin" : "bitcoin"; // Using bitcoin icon for assets
              } else if (route.name === "Pools") {
                iconName = focused ? "waves" : "waves-arrow-up"; // Using waves icon for pools
              }
              // You can return any component that you like here!
              return (
                <MaterialCommunityIcons
                  name={iconName}
                  size={size}
                  color={color}
                />
              );
            },
            tabBarActiveTintColor: theme.colors.primary,
            tabBarInactiveTintColor: theme.colors.onSurfaceVariant,
            // Optional: Style the tab bar itself
            // tabBarActiveBackgroundColor: theme.colors.secondaryContainer,
            // tabBarStyle: { backgroundColor: theme.colors.surfaceVariant },
          })}
        >
          <Tab.Screen
            name="Predict"
            component={PredictionStack}
            options={{ title: "Prediction" }}
          />
          <Tab.Screen
            name="Assets"
            component={AssetsScreen}
            options={{
              title: "Assets",
              headerShown: true, // Show header for simple screens if needed
              headerStyle: { backgroundColor: theme.colors.primary },
              headerTintColor: "#fff",
              headerTitleStyle: { fontWeight: "bold" },
            }}
          />
          <Tab.Screen
            name="Pools"
            component={PoolsScreen}
            options={{
              title: "Pools",
              headerShown: true, // Show header for simple screens if needed
              headerStyle: { backgroundColor: theme.colors.primary },
              headerTintColor: "#fff",
              headerTitleStyle: { fontWeight: "bold" },
            }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
