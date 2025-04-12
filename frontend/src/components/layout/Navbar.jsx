import React from 'react';
import { 
  Box, 
  Flex, 
  Text, 
  IconButton, 
  HStack, 
  Menu, 
  MenuButton, 
  MenuList, 
  MenuItem, 
  Avatar, 
  Button,
  useColorModeValue,
  Icon,
  Divider
} from '@chakra-ui/react';
import { FiMenu, FiBell, FiUser, FiLogOut, FiSettings, FiChevronDown } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';
import { useWeb3 } from '../../lib/web3-config';
import { useUI } from '../../lib/ui-context';

const Navbar = () => {
  const navigate = useNavigate();
  const { account, isConnected, connectWallet, disconnectWallet } = useWeb3();
  const { toggleSidebar, addNotification } = useUI();
  
  const bgColor = useColorModeValue('gray.800', 'gray.800');
  const borderColor = useColorModeValue('gray.700', 'gray.700');
  
  const handleConnect = async () => {
    try {
      await connectWallet();
      addNotification({
        title: 'Wallet Connected',
        message: 'Your wallet has been successfully connected.',
        type: 'success'
      });
    } catch (error) {
      addNotification({
        title: 'Connection Failed',
        message: error.message || 'Failed to connect wallet. Please try again.',
        type: 'error'
      });
    }
  };
  
  const handleDisconnect = () => {
    disconnectWallet();
    addNotification({
      title: 'Wallet Disconnected',
      message: 'Your wallet has been disconnected.',
      type: 'info'
    });
  };
  
  const formatAddress = (address) => {
    if (!address) return '';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  };
  
  return (
    <Box 
      as="header" 
      bg={bgColor} 
      borderBottom="1px" 
      borderColor={borderColor} 
      py={2} 
      px={4}
      position="sticky"
      top="0"
      zIndex="sticky"
    >
      <Flex justify="space-between" align="center">
        <HStack spacing={4}>
          <IconButton
            icon={<FiMenu />}
            variant="ghost"
            onClick={toggleSidebar}
            aria-label="Toggle Sidebar"
            color="gray.400"
            _hover={{ color: 'white', bg: 'gray.700' }}
          />
          <Box 
            as="a" 
            href="/" 
            fontSize="xl" 
            fontWeight="bold" 
            color="white"
            cursor="pointer"
            onClick={(e) => {
              e.preventDefault();
              navigate('/');
            }}
          >
            <Flex align="center">
              <Text 
                bgGradient="linear(to-r, brand.500, accent.500)" 
                bgClip="text"
                fontWeight="extrabold"
              >
                FLUXION
              </Text>
            </Flex>
          </Box>
        </HStack>
        
        <HStack spacing={4}>
          {isConnected ? (
            <Menu>
              <MenuButton
                as={Button}
                variant="outline"
                colorScheme="blue"
                rightIcon={<FiChevronDown />}
                size="sm"
              >
                {formatAddress(account)}
              </MenuButton>
              <MenuList bg="gray.800" borderColor="gray.700">
                <MenuItem 
                  icon={<FiUser />} 
                  onClick={() => navigate('/settings')}
                  _hover={{ bg: 'gray.700' }}
                  color="white"
                >
                  Profile
                </MenuItem>
                <MenuItem 
                  icon={<FiSettings />} 
                  onClick={() => navigate('/settings')}
                  _hover={{ bg: 'gray.700' }}
                  color="white"
                >
                  Settings
                </MenuItem>
                <Divider borderColor="gray.700" />
                <MenuItem 
                  icon={<FiLogOut />} 
                  onClick={handleDisconnect}
                  _hover={{ bg: 'gray.700' }}
                  color="white"
                >
                  Disconnect
                </MenuItem>
              </MenuList>
            </Menu>
          ) : (
            <Button 
              colorScheme="blue" 
              size="sm" 
              onClick={handleConnect}
            >
              Connect Wallet
            </Button>
          )}
          
          <IconButton
            icon={<FiBell />}
            variant="ghost"
            aria-label="Notifications"
            color="gray.400"
            _hover={{ color: 'white', bg: 'gray.700' }}
          />
          
          <Avatar 
            size="sm" 
            name="User" 
            bg="brand.500" 
            cursor="pointer"
            onClick={() => navigate('/settings')}
          />
        </HStack>
      </Flex>
    </Box>
  );
};

export default Navbar;
