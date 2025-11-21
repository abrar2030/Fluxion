import React from "react";
import { Box, Spinner, Text, VStack } from "@chakra-ui/react";

const LoadingOverlay = ({ message = "Loading..." }) => {
  return (
    <Box
      position="fixed"
      top={0}
      left={0}
      right={0}
      bottom={0}
      bg="blackAlpha.600"
      display="flex"
      alignItems="center"
      justifyContent="center"
      zIndex={9999}
    >
      <VStack spacing={4}>
        <Spinner
          thickness="4px"
          speed="0.65s"
          emptyColor="gray.200"
          color="blue.500"
          size="xl"
        />
        <Text color="white" fontSize="lg">
          {message}
        </Text>
      </VStack>
    </Box>
  );
};

export default LoadingOverlay;
