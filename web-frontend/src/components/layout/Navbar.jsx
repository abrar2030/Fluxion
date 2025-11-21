import React from "react";
import {
  Box,
  Flex,
  HStack,
  Image,
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
  Avatar,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  MenuDivider,
  IconButton,
  Text,
} from "@chakra-ui/react";
import {
  FiMenu,
  FiBell,
  FiChevronDown,
  FiUser,
  FiLogOut,
  FiSettings,
} from "react-icons/fi";
import { Link as RouterLink } from "react-router-dom";
import logo from "../../assets/images/fluxion-logo.svg";
import { useWeb3 } from "../../lib/web3-config.jsx";

const Navbar = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { account, isConnected, connectWallet } = useWeb3();

  // Format account address for display
  const formatAddress = (address) => {
    if (!address) return "";
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  };

  return (
    <Box>
      <Flex
        bg={useColorModeValue("gray.900", "gray.900")}
        color={useColorModeValue("white", "white")}
        minH={"60px"}
        py={{ base: 2 }}
        px={{ base: 4 }}
        borderBottom={1}
        borderStyle={"solid"}
        borderColor={useColorModeValue("gray.800", "gray.700")}
        align={"center"}
        position="fixed"
        top="0"
        width="100%"
        zIndex="1000"
        boxShadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
      >
        <Flex
          flex={{ base: 1, md: "auto" }}
          ml={{ base: -2 }}
          display={{ base: "flex", md: "none" }}
        >
          <IconButton
            onClick={onOpen}
            icon={<FiMenu />}
            variant={"outline"}
            aria-label={"Toggle Navigation"}
          />
        </Flex>

        <Flex flex={{ base: 1 }} justify={{ base: "center", md: "start" }}>
          <RouterLink to="/">
            <Image src={logo} alt="Fluxion Logo" h="10" />
          </RouterLink>
        </Flex>

        <HStack spacing={4} alignItems={"center"}>
          <IconButton
            size={"md"}
            variant={"ghost"}
            aria-label={"Notifications"}
            icon={<FiBell />}
            _hover={{
              bg: "brand.500",
              color: "white",
            }}
          />

          {isConnected ? (
            <Menu>
              <MenuButton
                as={Button}
                rounded={"full"}
                variant={"link"}
                cursor={"pointer"}
                minW={0}
              >
                <HStack>
                  <Avatar
                    size={"sm"}
                    bgGradient="linear(to-r, brand.500, accent.500)"
                  />
                  <Text display={{ base: "none", md: "flex" }}>
                    {formatAddress(account)}
                  </Text>
                  <Box display={{ base: "none", md: "flex" }}>
                    <FiChevronDown />
                  </Box>
                </HStack>
              </MenuButton>
              <MenuList
                bg={useColorModeValue("gray.800", "gray.700")}
                borderColor={useColorModeValue("gray.700", "gray.600")}
              >
                <MenuItem
                  icon={<FiUser />}
                  _hover={{
                    bg: "gray.700",
                  }}
                >
                  Profile
                </MenuItem>
                <MenuItem
                  icon={<FiSettings />}
                  _hover={{
                    bg: "gray.700",
                  }}
                >
                  Settings
                </MenuItem>
                <MenuDivider />
                <MenuItem
                  icon={<FiLogOut />}
                  _hover={{
                    bg: "gray.700",
                  }}
                >
                  Disconnect
                </MenuItem>
              </MenuList>
            </Menu>
          ) : (
            <Button
              size="md"
              variant="solid"
              colorScheme="brand"
              bgGradient="linear(to-r, brand.500, accent.500)"
              _hover={{
                bgGradient: "linear(to-r, brand.600, accent.600)",
                transform: "translateY(-2px)",
                boxShadow: "lg",
              }}
              onClick={connectWallet}
            >
              Connect Wallet
            </Button>
          )}
        </HStack>
      </Flex>

      <Drawer
        isOpen={isOpen}
        placement="left"
        onClose={onClose}
        returnFocusOnClose={false}
        onOverlayClick={onClose}
      >
        <DrawerOverlay />
        <DrawerContent
          bg={useColorModeValue("gray.900", "gray.900")}
          color="white"
        >
          <DrawerCloseButton />
          <DrawerHeader borderBottomWidth="1px">Menu</DrawerHeader>
          <DrawerBody>{/* Mobile menu content */}</DrawerBody>
        </DrawerContent>
      </Drawer>
    </Box>
  );
};

export default Navbar;
