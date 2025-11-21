import React from "react";
import { createContext, useContext, useState } from "react";

// Create context
const UIContext = createContext(null);

// Provider
export const UIProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState({});
  const [errors, setErrors] = useState({});

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Add notification
  const addNotification = (notification) => {
    const id = Date.now();
    const newNotification = {
      id,
      ...notification,
      timestamp: new Date(),
    };

    setNotifications([newNotification, ...notifications]);

    // Auto dismiss after timeout if not persistent
    if (!notification.persistent) {
      setTimeout(() => {
        dismissNotification(id);
      }, notification.duration || 5000);
    }

    return id;
  };

  // Dismiss notification
  const dismissNotification = (id) => {
    setNotifications(notifications.filter((n) => n.id !== id));
  };

  // Set loading state
  const setLoadingState = (key, isLoading) => {
    setLoading((prev) => ({
      ...prev,
      [key]: isLoading,
    }));
  };

  // Set error state
  const setErrorState = (key, error) => {
    setErrors((prev) => ({
      ...prev,
      [key]: error,
    }));

    if (error) {
      addNotification({
        title: "Error",
        message: error,
        type: "error",
        duration: 8000,
      });
    }
  };

  // Clear error state
  const clearErrorState = (key) => {
    setErrors((prev) => {
      const newErrors = { ...prev };
      delete newErrors[key];
      return newErrors;
    });
  };

  // Context value
  const value = {
    darkMode,
    toggleDarkMode,
    sidebarOpen,
    toggleSidebar,
    notifications,
    addNotification,
    dismissNotification,
    loading,
    setLoadingState,
    errors,
    setErrorState,
    clearErrorState,
  };

  return <UIContext.Provider value={value}>{children}</UIContext.Provider>;
};

// Custom hook to use the UI context
export const useUI = () => {
  const context = useContext(UIContext);
  if (!context) {
    throw new Error("useUI must be used within a UIProvider");
  }
  return context;
};

export default useUI;
