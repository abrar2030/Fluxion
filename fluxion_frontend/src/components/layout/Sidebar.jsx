import React from 'react';
import { 
  Box, 
  VStack, 
  Icon, 
  Flex, 
  Text, 
  Divider,
  useColorModeValue
} from '@chakra-ui/react';
import { 
  FiHome, 
  FiDroplet, 
  FiPlusCircle, 
  FiBarChart2, 
  FiPackage, 
  FiSettings 
} from 'react-icons/fi';
import { useNavigate, useLocation } from 'react-router-dom';

const SidebarItem = ({ icon, label, path, isActive }) => {
  const navigate = useNavigate();
  const activeBg = useColorModeValue('gray.700', 'gray.700');
  const hoverBg = useColorModeValue('gray.700', 'gray.700');
  const activeColor = useColorModeValue('brand.500', 'brand.400');
  
  return (
    <Flex
      align="center"
      p="4"
      mx="4"
      borderRadius="lg"
      role="group"
      cursor="pointer"
      bg={isActive ? activeBg : 'transparent'}
      color={isActive ? activeColor : 'gray.400'}
      _hover={{
        bg: hoverBg,
        color: 'white',
      }}
      onClick={() => navigate(path)}
      className={isActive ? 'active-nav-item' : ''}
    >
      <Icon
        mr="4"
        fontSize="16"
        as={icon}
      />
      <Text>{label}</Text>
      {isActive && (
        <Box
          position="absolute"
          left="0"
          width="4px"
          height="32px"
          bg="brand.500"
          borderRightRadius="md"
        />
      )}
    </Flex>
  );
};

const Sidebar = ({ isOpen }) => {
  const location = useLocation();
  const bgColor = useColorModeValue('gray.800', 'gray.800');
  const borderColor = useColorModeValue('gray.700', 'gray.700');
  
  const menuItems = [
    { icon: FiHome, label: 'Dashboard', path: '/' },
    { icon: FiDroplet, label: 'Pools', path: '/pools' },
    { icon: FiPlusCircle, label: 'Create Pool', path: '/create-pool' },
    { icon: FiBarChart2, label: 'Analytics', path: '/analytics' },
    { icon: FiPackage, label: 'Synthetics', path: '/synthetics' },
    { icon: FiSettings, label: 'Settings', path: '/settings' },
  ];
  
  return (
    <Box
      position="fixed"
      left="0"
      h="calc(100vh - 60px)"
      w={{ base: 'full', md: '60px' }}
      bg={bgColor}
      borderRight="1px"
      borderColor={borderColor}
      transform={{ base: isOpen ? 'translateX(0)' : 'translateX(-100%)', md: 'translateX(0)' }}
      transition="transform 0.3s ease"
      zIndex="sidebar"
      className="sidebar"
    >
      <VStack spacing={0} align="stretch" pt={4}>
        {menuItems.map((item, index) => (
          <SidebarItem
            key={index}
            icon={item.icon}
            label={item.label}
            path={item.path}
            isActive={
              item.path === '/' 
                ? location.pathname === '/' 
                : location.pathname.startsWith(item.path)
            }
          />
        ))}
      </VStack>
    </Box>
  );
};

export default Sidebar;
