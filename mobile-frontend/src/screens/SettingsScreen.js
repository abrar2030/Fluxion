import React, { useState } from 'react';
import { Box, Text, VStack, Heading, ScrollView, Card, HStack, Switch, useColorMode, Divider, Button, Icon } from 'native-base';
import { Ionicons } from '@expo/vector-icons';

const SettingsScreen = () => {
  const { colorMode, toggleColorMode } = useColorMode(); // NativeBase color mode hook
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [faceIdEnabled, setFaceIdEnabled] = useState(false);

  const handleLogout = () => {
    // Implement logout logic here
    console.log('User logged out');
  };

  return (
    <ScrollView flex={1} bg="gray.950">
      <VStack space={4} p={4} alignItems="stretch">
        <Heading color="white" textAlign="center">Settings</Heading>

        {/* Appearance Settings */}
        <Card bg="gray.800" p={4} rounded="lg">
          <Heading size="sm" color="white" mb={3}>Appearance</Heading>
          <HStack justifyContent="space-between" alignItems="center">
            <Text color="white">Dark Mode</Text>
            <Switch
              isChecked={colorMode === 'dark'}
              onToggle={toggleColorMode} // Use NativeBase's toggle
              colorScheme="primary"
            />
          </HStack>
        </Card>

        {/* Notification Settings */}
        <Card bg="gray.800" p={4} rounded="lg">
          <Heading size="sm" color="white" mb={3}>Notifications</Heading>
          <HStack justifyContent="space-between" alignItems="center">
            <Text color="white">Enable Push Notifications</Text>
            <Switch
              isChecked={notificationsEnabled}
              onToggle={() => setNotificationsEnabled(!notificationsEnabled)}
              colorScheme="primary"
            />
          </HStack>
          {/* Add more notification options if needed */}
        </Card>

        {/* Security Settings */}
        <Card bg="gray.800" p={4} rounded="lg">
          <Heading size="sm" color="white" mb={3}>Security</Heading>
          <HStack justifyContent="space-between" alignItems="center" mb={3}>
            <Text color="white">Enable Face ID / Biometrics</Text>
            <Switch
              isChecked={faceIdEnabled}
              onToggle={() => setFaceIdEnabled(!faceIdEnabled)}
              colorScheme="primary"
            />
          </HStack>
          <Divider my={2} bg="gray.700" />
          <Button variant="outline" colorScheme="blueGray" onPress={() => console.log('Change Password')}>
            Change Password
          </Button>
        </Card>

        {/* Account Actions */}
        <Card bg="gray.800" p={4} rounded="lg">
           <Heading size="sm" color="white" mb={3}>Account</Heading>
           <Button
             colorScheme="red"
             leftIcon={<Icon as={Ionicons} name="log-out-outline" size="sm" />}
             onPress={handleLogout}
           >
             Logout
           </Button>
        </Card>

      </VStack>
    </ScrollView>
  );
};

export default SettingsScreen;

