import React, { useState } from "react";
import {
  View,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Platform,
} from "react-native";
import {
  TextInput,
  Button,
  HelperText,
  useTheme,
  Card,
  Text,
  Chip,
  IconButton,
} from "react-native-paper";
import DateTimePickerModal from "react-native-modal-datetime-picker";
import { format } from "date-fns"; // Using date-fns for reliable formatting

const PredictionForm = ({ onSubmit, isLoading }) => {
  const [selectedTimestamps, setSelectedTimestamps] = useState([]);
  const [meterIds, setMeterIds] = useState("");
  const [contextFeatures, setContextFeatures] = useState("");
  const [errors, setErrors] = useState({});
  const [isDatePickerVisible, setDatePickerVisibility] = useState(false);
  const theme = useTheme();

  const showDatePicker = () => {
    setDatePickerVisibility(true);
  };

  const hideDatePicker = () => {
    setDatePickerVisibility(false);
  };

  const handleConfirm = (date) => {
    // Add the selected date and sort them
    setSelectedTimestamps((prev) => [...prev, date].sort((a, b) => a - b));
    hideDatePicker();
    // Clear timestamp error if present
    if (errors.timestamps) {
      setErrors((prev) => ({ ...prev, timestamps: null }));
    }
  };

  const removeTimestamp = (indexToRemove) => {
    setSelectedTimestamps((prev) =>
      prev.filter((_, index) => index !== indexToRemove),
    );
  };

  const validateInput = () => {
    const newErrors = {};
    if (selectedTimestamps.length === 0)
      newErrors.timestamps = "At least one timestamp is required.";
    if (!meterIds.trim()) newErrors.meterIds = "Meter IDs are required.";
    if (!contextFeatures.trim()) {
      newErrors.contextFeatures = "Context Features are required.";
    } else {
      try {
        JSON.parse(contextFeatures);
      } catch (e) {
        newErrors.contextFeatures = "Context Features must be valid JSON.";
      }
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (!validateInput()) {
      return;
    }

    try {
      // Format timestamps to ISO 8601 strings
      const timestampArray = selectedTimestamps.map((date) =>
        format(date, "yyyy-MM-dd'T'HH:mm:ss"),
      );
      const meterIdArray = meterIds
        .split(",")
        .map((id) => id.trim())
        .filter((id) => id); // Filter empty strings
      const contextFeaturesObj = JSON.parse(contextFeatures);
      onSubmit(timestampArray, meterIdArray, contextFeaturesObj);
    } catch (error) {
      setErrors({
        form:
          error.message || "An unexpected error occurred during submission.",
      });
    }
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
    >
      <Card style={styles.card}>
        <Card.Content>
          {/* Timestamp Input */}
          <View style={styles.timestampContainer}>
            <Text style={styles.label}>Timestamps</Text>
            <Button
              icon="calendar-plus"
              mode="outlined"
              onPress={showDatePicker}
              style={styles.addButton}
            >
              Add Timestamp
            </Button>
          </View>
          <View style={styles.chipContainer}>
            {selectedTimestamps.map((date, index) => (
              <Chip
                key={index}
                icon="clock"
                onClose={() => removeTimestamp(index)}
                style={styles.chip}
                mode="outlined"
              >
                {format(date, "yyyy-MM-dd HH:mm")}
              </Chip>
            ))}
          </View>
          <HelperText type="error" visible={!!errors.timestamps}>
            {errors.timestamps}
          </HelperText>

          <DateTimePickerModal
            isVisible={isDatePickerVisible}
            mode="datetime"
            onConfirm={handleConfirm}
            onCancel={hideDatePicker}
          />

          {/* Meter IDs Input */}
          <TextInput
            label="Meter IDs (comma-separated)"
            value={meterIds}
            onChangeText={setMeterIds}
            placeholder="meter1, meter2, meter3"
            mode="outlined"
            style={styles.input}
            error={!!errors.meterIds}
          />
          <HelperText type="error" visible={!!errors.meterIds}>
            {errors.meterIds}
          </HelperText>

          {/* Context Features Input */}
          <TextInput
            label="Context Features (JSON format)"
            value={contextFeatures}
            onChangeText={setContextFeatures}
            placeholder='{"temperature": 22.5, "humidity": 65}'
            mode="outlined"
            style={[styles.input, styles.jsonInput]}
            multiline
            numberOfLines={4}
            error={!!errors.contextFeatures}
          />
          <HelperText type="error" visible={!!errors.contextFeatures}>
            {errors.contextFeatures}
          </HelperText>

          {/* Submit Button */}
          <Button
            mode="contained"
            onPress={handleSubmit}
            disabled={isLoading}
            loading={isLoading}
            style={styles.button}
            labelStyle={styles.buttonLabel}
          >
            {isLoading ? "Getting Prediction..." : "Get Prediction"}
          </Button>
          <HelperText type="error" visible={!!errors.form}>
            {errors.form}
          </HelperText>
        </Card.Content>
      </Card>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
  },
  card: {
    marginBottom: 16,
  },
  label: {
    fontSize: 16,
    marginBottom: 8,
    color: "#333", // Adjust color as needed
    fontWeight: "bold",
  },
  timestampContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  addButton: {
    // Style for the add button if needed
  },
  chipContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginBottom: 8,
  },
  chip: {
    marginRight: 8,
    marginBottom: 8,
  },
  input: {
    marginBottom: 8,
  },
  jsonInput: {
    height: 120,
    textAlignVertical: "top",
  },
  button: {
    marginTop: 16,
    paddingVertical: 8,
  },
  buttonLabel: {
    fontSize: 16,
  },
});

export default PredictionForm;
