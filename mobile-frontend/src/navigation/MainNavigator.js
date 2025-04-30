import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons'; // Using Expo's vector icons
import { useTheme } from 'native-base';

// Import Screens
import DashboardScreen from '../screens/DashboardScreen';
import PoolsScreen from '../screens/PoolsScreen';
import SyntheticsScreen from '../screens/SyntheticsScreen';
import AnalyticsScreen from '../screens/AnalyticsScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Tab = createBottomTabNavigator();

const MainNavigator = () => {
  const theme = useTheme();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Dashboard') {
            iconName = focused ? 'ios-home' : 'ios-home-outline';
          } else if (route.name === 'Pools') {
            iconName = focused ? 'ios-water' : 'ios-water-outline';
          } else if (route.name === 'Synthetics') {
            iconName = focused ? 'ios-swap-horizontal' : 'ios-swap-horizontal-outline';
          } else if (route.name === 'Analytics') {
            iconName = focused ? 'ios-analytics' : 'ios-analytics-outline';
          } else if (route.name === 'Settings') {
            iconName = focused ? 'ios-settings' : 'ios-settings-outline';
          }

          // You can return any component that you like here!
          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary[500],
        tabBarInactiveTintColor: theme.colors.gray[500],
        tabBarStyle: {
          backgroundColor: theme.colors.gray[900], // Dark background for tab bar
          borderTopColor: theme.colors.gray[700], // Subtle border
        },
        headerStyle: {
          backgroundColor: theme.colors.gray[900], // Dark background for header
        },
        headerTintColor: theme.colors.gray[100], // Light text for header
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      })}
    >
      <Tab.Screen name="Dashboard" component={DashboardScreen} />
      <Tab.Screen name="Pools" component={PoolsScreen} />
      <Tab.Screen name="Synthetics" component={SyntheticsScreen} />
      <Tab.Screen name="Analytics" component={AnalyticsScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
};

export default MainNavigator;

