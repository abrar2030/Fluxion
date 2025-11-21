import React from "react";
import {
  Box,
  Flex,
  Text,
  Spinner,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
} from "@chakra-ui/react";

const LoadingOverlay = ({ message = "Loading..." }) => {
  return (
    <Flex
      position="fixed"
      top="0"
      left="0"
      right="0"
      bottom="0"
      bg="rgba(0, 0, 0, 0.7)"
      zIndex="9999"
      justify="center"
      align="center"
      direction="column"
    >
      <Spinner
        thickness="4px"
        speed="0.65s"
        emptyColor="gray.200"
        color="var(--color-primary)"
        size="xl"
        mb={4}
      />
      <Text color="white" fontSize="lg" fontWeight="medium">
        {message}
      </Text>
    </Flex>
  );
};

const ErrorBoundary = ({ children }) => {
  const [hasError, setHasError] = React.useState(false);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    const errorHandler = (error) => {
      console.error("Application error:", error);
      setError(error.message || "An unexpected error occurred");
      setHasError(true);
    };

    window.addEventListener("error", errorHandler);
    return () => window.removeEventListener("error", errorHandler);
  }, []);

  if (hasError) {
    return (
      <Box p={8} maxW="800px" mx="auto" mt={8}>
        <Alert
          status="error"
          variant="subtle"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          textAlign="center"
          height="auto"
          py={6}
          borderRadius="md"
        >
          <AlertIcon boxSize="40px" mr={0} />
          <AlertTitle mt={4} mb={1} fontSize="lg">
            Application Error
          </AlertTitle>
          <AlertDescription maxWidth="sm">
            {error ||
              "An unexpected error occurred. Please refresh the page and try again."}
          </AlertDescription>
        </Alert>
      </Box>
    );
  }

  return children;
};

const NotificationCenter = ({ notifications }) => {
  return (
    <Box
      position="fixed"
      bottom="20px"
      right="20px"
      zIndex="1000"
      maxW="400px"
      w="100%"
    >
      {notifications.map((notification) => (
        <Alert
          key={notification.id}
          status={notification.type || "info"}
          variant="solid"
          mb={3}
          borderRadius="md"
          boxShadow="lg"
          className="slide-in"
        >
          <AlertIcon />
          <Box flex="1">
            {notification.title && (
              <AlertTitle>{notification.title}</AlertTitle>
            )}
            <AlertDescription>{notification.message}</AlertDescription>
          </Box>
        </Alert>
      ))}
    </Box>
  );
};

export { LoadingOverlay, ErrorBoundary, NotificationCenter };
export default { LoadingOverlay, ErrorBoundary, NotificationCenter };
