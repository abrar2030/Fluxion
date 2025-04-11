import React from 'react';
import { 
  Box, 
  Flex, 
  Text, 
  Link, 
  HStack,
  useColorModeValue 
} from '@chakra-ui/react';

const Footer = () => {
  const bgColor = useColorModeValue('gray.800', 'gray.800');
  const borderColor = useColorModeValue('gray.700', 'gray.700');
  
  return (
    <Box 
      as="footer" 
      bg={bgColor} 
      borderTop="1px" 
      borderColor={borderColor} 
      py={3} 
      px={4}
    >
      <Flex 
        justify="space-between" 
        align="center" 
        maxW="7xl" 
        mx="auto"
        flexDirection={{ base: 'column', md: 'row' }}
        gap={{ base: 2, md: 0 }}
      >
        <Text color="gray.400" fontSize="sm">
          © {new Date().getFullYear()} Fluxion. All rights reserved.
        </Text>
        
        <HStack spacing={4}>
          <Link href="#" color="gray.400" fontSize="sm" _hover={{ color: 'white' }}>
            Terms of Service
          </Link>
          <Link href="#" color="gray.400" fontSize="sm" _hover={{ color: 'white' }}>
            Privacy Policy
          </Link>
          <Link href="#" color="gray.400" fontSize="sm" _hover={{ color: 'white' }}>
            Documentation
          </Link>
        </HStack>
      </Flex>
    </Box>
  );
};

export default Footer;
