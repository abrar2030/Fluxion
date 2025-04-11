import React, { useState } from 'react';
import {
  Box,
  Heading,
  SimpleGrid,
  Card,
  CardBody,
  Text,
  Button,
  Flex,
  HStack,
  VStack,
  Input,
  FormControl,
  FormLabel,
  Select,
  Switch,
  Divider,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Avatar,
  AvatarBadge,
  Icon,
  InputGroup,
  InputRightElement,
  useColorModeValue,
  useToast
} from '@chakra-ui/react';
import { 
  FiSettings, 
  FiUser, 
  FiShield, 
  FiGlobe, 
  FiEye, 
  FiEyeOff, 
  FiSave,
  FiAlertTriangle,
  FiBell,
  FiMoon,
  FiSun
} from 'react-icons/fi';

const Settings = () => {
  const toast = useToast();
  const [showPassword, setShowPassword] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  
  const cardBg = useColorModeValue('gray.700', 'gray.700');
  const borderColor = useColorModeValue('gray.600', 'gray.600');

  const handleSaveSettings = (section) => {
    toast({
      title: "Settings saved",
      description: `Your ${section} settings have been updated`,
      status: "success",
      duration: 3000,
      isClosable: true,
    });
  };

  return (
    <Box maxW="7xl" mx="auto" pt={5} px={{ base: 2, sm: 12, md: 17 }} className="fade-in">
      <Heading as="h1" mb={6} fontSize="3xl" color="white">
        Settings
      </Heading>
      
      <Tabs variant="soft-rounded" colorScheme="blue" mb={8}>
        <TabList mb={4}>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>
            <Icon as={FiUser} mr={2} />
            Profile
          </Tab>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>
            <Icon as={FiShield} mr={2} />
            Security
          </Tab>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>
            <Icon as={FiGlobe} mr={2} />
            Preferences
          </Tab>
          <Tab color="gray.300" _selected={{ color: 'white', bg: 'brand.500' }}>
            <Icon as={FiBell} mr={2} />
            Notifications
          </Tab>
        </TabList>
        
        <TabPanels>
          {/* Profile Tab */}
          <TabPanel p={0}>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
              <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
                <CardBody>
                  <Heading size="md" mb={6} color="white">User Profile</Heading>
                  
                  <Flex direction={{ base: 'column', sm: 'row' }} mb={6} align="center">
                    <Avatar 
                      size="xl" 
                      name="User Name" 
                      bg="brand.500"
                      mr={{ base: 0, sm: 6 }}
                      mb={{ base: 4, sm: 0 }}
                    >
                      <AvatarBadge boxSize="1.25em" bg="green.500" />
                    </Avatar>
                    
                    <VStack spacing={2} align={{ base: 'center', sm: 'flex-start' }}>
                      <Text color="white" fontWeight="bold" fontSize="xl">User Name</Text>
                      <Text color="gray.300">Connected: 0x1234...5678</Text>
                      <Button size="sm" colorScheme="blue" variant="outline">
                        Change Avatar
                      </Button>
                    </VStack>
                  </Flex>
                  
                  <VStack spacing={4} align="stretch">
                    <FormControl>
                      <FormLabel color="gray.300">Display Name</FormLabel>
                      <Input 
                        defaultValue="User Name" 
                        bg="gray.800"
                        color="white"
                        borderColor="gray.600"
                      />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel color="gray.300">Email Address</FormLabel>
                      <Input 
                        defaultValue="user@example.com" 
                        bg="gray.800"
                        color="white"
                        borderColor="gray.600"
                      />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel color="gray.300">Bio</FormLabel>
                      <Input 
                        as="textarea"
                        rows={3}
                        placeholder="Tell us about yourself" 
                        bg="gray.800"
                        color="white"
                        borderColor="gray.600"
                      />
                    </FormControl>
                    
                    <Button 
                      colorScheme="blue" 
                      leftIcon={<FiSave />}
                      onClick={() => handleSaveSettings('profile')}
                    >
                      Save Profile
                    </Button>
                  </VStack>
                </CardBody>
              </Card>
              
              <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
                <CardBody>
                  <Heading size="md" mb={6} color="white">Wallet Information</Heading>
                  
                  <VStack spacing={4} align="stretch">
                    <Flex justify="space-between" align="center">
                      <Text color="gray.300">Connected Wallet:</Text>
                      <Text color="white" fontWeight="bold">MetaMask</Text>
                    </Flex>
                    
                    <Flex justify="space-between" align="center">
                      <Text color="gray.300">Address:</Text>
                      <Text color="white" fontWeight="bold">0x1234...5678</Text>
                    </Flex>
                    
                    <Flex justify="space-between" align="center">
                      <Text color="gray.300">Network:</Text>
                      <Text color="white" fontWeight="bold">Ethereum Mainnet</Text>
                    </Flex>
                    
                    <Flex justify="space-between" align="center">
                      <Text color="gray.300">Connection Status:</Text>
                      <Text color="green.400" fontWeight="bold">Connected</Text>
                    </Flex>
                    
                    <Divider borderColor="gray.600" my={2} />
                    
                    <FormControl>
                      <FormLabel color="gray.300">Default Gas Price</FormLabel>
                      <Select 
                        defaultValue="medium" 
                        bg="gray.800"
                        color="white"
                        borderColor="gray.600"
                      >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                      </Select>
                    </FormControl>
                    
                    <Button colorScheme="blue" variant="outline">
                      Disconnect Wallet
                    </Button>
                  </VStack>
                </CardBody>
              </Card>
            </SimpleGrid>
          </TabPanel>
          
          {/* Security Tab */}
          <TabPanel p={0}>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
              <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
                <CardBody>
                  <Heading size="md" mb={6} color="white">Account Security</Heading>
                  
                  <VStack spacing={4} align="stretch">
                    <FormControl>
                      <FormLabel color="gray.300">Current Password</FormLabel>
                      <InputGroup>
                        <Input 
                          type={showPassword ? 'text' : 'password'} 
                          placeholder="Enter current password" 
                          bg="gray.800"
                          color="white"
                          borderColor="gray.600"
                        />
                        <InputRightElement>
                          <Button 
                            variant="ghost" 
                            onClick={() => setShowPassword(!showPassword)}
                            color="gray.400"
                            _hover={{ color: 'white' }}
                          >
                            <Icon as={showPassword ? FiEyeOff : FiEye} />
                          </Button>
                        </InputRightElement>
                      </InputGroup>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel color="gray.300">New Password</FormLabel>
                      <InputGroup>
                        <Input 
                          type={showPassword ? 'text' : 'password'} 
                          placeholder="Enter new password" 
                          bg="gray.800"
                          color="white"
                          borderColor="gray.600"
                        />
                        <InputRightElement>
                          <Button 
                            variant="ghost" 
                            onClick={() => setShowPassword(!showPassword)}
                            color="gray.400"
                            _hover={{ color: 'white' }}
                          >
                            <Icon as={showPassword ? FiEyeOff : FiEye} />
                          </Button>
                        </InputRightElement>
                      </InputGroup>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel color="gray.300">Confirm New Password</FormLabel>
                      <InputGroup>
                        <Input 
                          type={showPassword ? 'text' : 'password'} 
                          placeholder="Confirm new password" 
                          bg="gray.800"
                          color="white"
                          borderColor="gray.600"
                        />
                        <InputRightElement>
                          <Button 
                            variant="ghost" 
                            onClick={() => setShowPassword(!showPassword)}
                            color="gray.400"
                            _hover={{ color: 'white' }}
                          >
                            <Icon as={showPassword ? FiEyeOff : FiEye} />
                          </Button>
                        </InputRightElement>
                      </InputGroup>
                    </FormControl>
                    
                    <Button 
                      colorScheme="blue" 
                      leftIcon={<FiSave />}
                      onClick={() => handleSaveSettings('password')}
                    >
                      Update Password
                    </Button>
                  </VStack>
                </CardBody>
              </Card>
              
              <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
                <CardBody>
                  <Heading size="md" mb={6} color="white">Two-Factor Authentication</Heading>
                  
                  <VStack spacing={4} align="stretch">
                    <Flex align="center" justify="space-between">
                      <Text color="white">Enable 2FA</Text>
                      <Switch colorScheme="blue" size="lg" />
                    </Flex>
                    
                    <Text color="gray.300">
                      Two-factor authentication adds an extra layer of security to your account. 
                      In addition to your password, you'll need to enter a code from your phone.
                    </Text>
                    
                    <Divider borderColor="gray.600" my={2} />
                    
                    <Heading size="sm" color="white" mb={2}>Security Activity</Heading>
                    
                    <VStack spacing={3} align="stretch">
                      {[
                        { action: 'Password changed', date: '2 weeks ago', ip: '192.168.1.1' },
                        { action: 'New login', date: '3 days ago', ip: '192.168.1.1' },
                        { action: 'New device added', date: '1 day ago', ip: '192.168.1.1' },
                      ].map((activity, index) => (
                        <Flex 
                          key={index} 
                          justify="space-between" 
                          p={3} 
                          borderRadius="md" 
                          bg="gray.800"
                        >
                          <VStack align="start" spacing={0}>
                            <Text color="white">{activity.action}</Text>
                            <Text color="gray.400" fontSize="sm">{activity.date}</Text>
                          </VStack>
                          <Text color="gray.400" fontSize="sm">IP: {activity.ip}</Text>
                        </Flex>
                      ))}
                    </VStack>
                  </VStack>
                </CardBody>
              </Card>
            </SimpleGrid>
          </TabPanel>
          
          {/* Preferences Tab */}
          <TabPanel p={0}>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
              <Card bg={cardBg} borderColor={borderColor} borderWidth="1px" borderRadius="lg" className="slide-up">
                <CardBody>
                  <Heading size="md" mb={6} color="white">Display Settings</Heading>
                  
                  <VStack spacing={4} align="stretch">
                    <Flex align="center" justify="space-between">
                      <HStack>
                        <Icon as={darkMode ? FiMoon : FiSun} color="white" />
                        <Text color="white">Dark Mode</Text>
                      </HStack>
                      <Switch 
                        colorScheme="blue" 
                        size="lg" 
                        isChecked={darkMode}
                        onChange={() => setDarkMode(!darkMode)}
                      />
                    </Flex>
                    
                    <FormControl>
                      <FormLabel color="gray.300">Language</FormLabel>
                      <Select 
                        defaultValue="en" 
                        bg="gray.800"
                        color="white"
                        borderColor="gray.600"
                      >
                        <option value="en">English</option>
                        <option value="es">Spanish</option>
                        <option value="fr">French</option>
                        <option value="de">German</option>
                        <option value="zh">Chinese</option>
                        <option value="ja">Japanese</option>
                      </Select>
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel color="gray.300">Currency Display</FormLabel>
                      <Select 
                        defaultValue="usd" 
                        bg="gray.800"
                        color="white"
                        borderColor="gray.600"
                      >
                        <option value="usd">USD ($)</option>
                        <option value="eur">EUR (€)</option>
                        <option value="gbp">GBP (£)</option>
                        <option value="jpy">JPY (¥)</option>
                        <option value="cny">CNY (¥)</option>
                      </Select>
  
(Content truncated due to size limit. Use line ranges to read in chunks)