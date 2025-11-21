import React from "react";
import {
  Box,
  Flex,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  useColorModeValue,
  Icon,
  useDisclosure,
  Drawer,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  DrawerHeader,
  DrawerBody,
  Image,
} from "@chakra-ui/react";
import {
  FiMenu,
  FiHome,
  FiDroplet,
  FiBarChart2,
  FiSettings,
  FiDollarSign,
  FiExternalLink,
} from "react-icons/fi";
import { Link as RouterLink } from "react-router-dom";
import logo from "../../assets/images/fluxion-logo.svg";

const NavItem = ({ icon, children, to, ...rest }) => {
  return (
    <Box
      as={RouterLink}
      to={to}
      style={{ textDecoration: "none" }}
      _focus={{ boxShadow: "none" }}
    >
      <Flex
        align="center"
        p="4"
        mx="4"
        borderRadius="lg"
        role="group"
        cursor="pointer"
        _hover={{
          bg: "brand.500",
          color: "white",
          transform: "translateY(-2px)",
          transition: "all 0.3s ease",
        }}
        {...rest}
      >
        {icon && (
          <Icon
            mr="4"
            fontSize="16"
            _groupHover={{
              color: "white",
            }}
            as={icon}
          />
        )}
        {children}
      </Flex>
    </Box>
  );
};

const SidebarContent = ({ ...rest }) => {
  return (
    <Box
      bg={useColorModeValue("gray.900", "gray.900")}
      borderRight="1px"
      borderRightColor={useColorModeValue("gray.800", "gray.700")}
      w={{ base: "full", md: 60 }}
      pos="fixed"
      h="full"
      {...rest}
    >
      <Flex h="20" alignItems="center" mx="8" justifyContent="space-between">
        <Image src={logo} alt="Fluxion Logo" h="10" />
        <Heading
          as="h1"
          fontSize="xl"
          fontFamily="heading"
          letterSpacing="tight"
        >
          Fluxion
        </Heading>
      </Flex>
      <VStack spacing={1} align="stretch" mt={6}>
        <NavItem icon={FiHome} to="/">
          Dashboard
        </NavItem>
        <NavItem icon={FiDroplet} to="/pools">
          Liquidity Pools
        </NavItem>
        <NavItem icon={FiDollarSign} to="/synthetics">
          Synthetic Assets
        </NavItem>
        <NavItem icon={FiBarChart2} to="/analytics">
          Analytics
        </NavItem>
        <NavItem icon={FiSettings} to="/settings">
          Settings
        </NavItem>
      </VStack>

      <Box position="absolute" bottom="5" width="100%">
        <Box mx="8" mb="4">
          <Button
            leftIcon={<FiExternalLink />}
            colorScheme="brand"
            variant="outline"
            size="sm"
            width="full"
            _hover={{
              bg: "brand.500",
              color: "white",
              borderColor: "brand.500",
            }}
          >
            Documentation
          </Button>
        </Box>
        <Text fontSize="xs" color="gray.400" textAlign="center" px="8">
          Fluxion v1.0.0
        </Text>
      </Box>
    </Box>
  );
};

const MobileNav = ({ onOpen, ...rest }) => {
  return (
    <Flex
      ml={{ base: 0, md: 60 }}
      px={{ base: 4, md: 4 }}
      height="20"
      alignItems="center"
      bg={useColorModeValue("gray.900", "gray.900")}
      borderBottomWidth="1px"
      borderBottomColor={useColorModeValue("gray.800", "gray.700")}
      justifyContent={{ base: "space-between", md: "flex-end" }}
      {...rest}
    >
      <Box display={{ base: "flex", md: "none" }}>
        <Button
          variant="outline"
          onClick={onOpen}
          aria-label="open menu"
          leftIcon={<FiMenu />}
        >
          Menu
        </Button>
      </Box>

      <Box display={{ base: "flex", md: "none" }}>
        <Image src={logo} alt="Fluxion Logo" h="8" />
      </Box>

      <HStack spacing={{ base: "0", md: "6" }}>
        <Button
          size="sm"
          variant="solid"
          colorScheme="brand"
          bgGradient="linear(to-r, brand.500, accent.500)"
          _hover={{
            bgGradient: "linear(to-r, brand.600, accent.600)",
            transform: "translateY(-2px)",
            boxShadow: "lg",
          }}
        >
          Connect Wallet
        </Button>
      </HStack>
    </Flex>
  );
};

const Sidebar = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <Box minH="100vh" bg={useColorModeValue("gray.950", "gray.900")}>
      <SidebarContent display={{ base: "none", md: "block" }} />
      <Drawer
        isOpen={isOpen}
        placement="left"
        onClose={onClose}
        returnFocusOnClose={false}
        onOverlayClick={onClose}
        size="full"
      >
        <DrawerOverlay />
        <DrawerContent>
          <DrawerCloseButton />
          <DrawerHeader borderBottomWidth="1px">Fluxion Menu</DrawerHeader>
          <DrawerBody>
            <SidebarContent w="full" borderRight="none" />
          </DrawerBody>
        </DrawerContent>
      </Drawer>
      <MobileNav onOpen={onOpen} />
      <Box ml={{ base: 0, md: 60 }} p="4">
        {/* Content */}
      </Box>
    </Box>
  );
};

export default Sidebar;
