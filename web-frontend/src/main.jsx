import React from "react";
import ReactDOM from "react-dom/client";
import { ChakraProvider, extendTheme } from "@chakra-ui/react";
import { BrowserRouter as Router } from "react-router-dom";
import App from "./App";
import { Web3Provider } from "./lib/web3-config.jsx";
import "./index.css";

// Theme customization
const theme = extendTheme({
  colors: {
    brand: {
      50: "#e6f7ff",
      100: "#b3e0ff",
      200: "#80caff",
      300: "#4db3ff",
      400: "#1a9dff",
      500: "#0080ff",
      600: "#0066cc",
      700: "#004d99",
      800: "#003366",
      900: "#001a33",
    },
    accent: {
      50: "#fff0e6",
      100: "#ffd6b3",
      200: "#ffbd80",
      300: "#ffa34d",
      400: "#ff8a1a",
      500: "#ff7000",
      600: "#cc5a00",
      700: "#994300",
      800: "#662d00",
      900: "#331600",
    },
    gray: {
      50: "#f9fafb",
      100: "#f3f4f6",
      200: "#e5e7eb",
      300: "#d1d5db",
      400: "#9ca3af",
      500: "#6b7280",
      600: "#4b5563",
      700: "#374151",
      800: "#1f2937",
      900: "#111827",
      950: "#0d1117",
    },
  },
  fonts: {
    body: "Inter, system-ui, sans-serif",
    heading: "Inter, system-ui, sans-serif",
  },
  styles: {
    global: (props) => ({
      body: {
        bg: "gray.900",
        color: "white",
      },
    }),
  },
  components: {
    Button: {
      baseStyle: {
        fontWeight: "semibold",
        borderRadius: "md",
      },
      variants: {
        solid: (props) => ({
          bg: props.colorScheme === "brand" ? "brand.500" : undefined,
          _hover: {
            bg: props.colorScheme === "brand" ? "brand.600" : undefined,
            transform: "translateY(-2px)",
            boxShadow: "lg",
          },
          transition: "all 0.2s ease-in-out",
        }),
      },
    },
    Card: {
      baseStyle: {
        container: {
          borderRadius: "lg",
          overflow: "hidden",
        },
      },
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ChakraProvider theme={theme}>
      <Web3Provider>
        <Router>
          <App />
        </Router>
      </Web3Provider>
    </ChakraProvider>
  </React.StrictMode>,
);
