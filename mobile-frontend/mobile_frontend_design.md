## Fluxora Mobile Frontend Design

### 1. Overview

This document outlines the design for a modern mobile frontend for the Fluxora energy prediction project. The application will target both Android and iOS platforms and provide an interface to interact with the existing Fluxora backend API.

### 2. Technology Stack

*   **Framework:** React Native with Expo
    *   Rationale: Enables cross-platform development (Android, iOS, Web) from a single codebase, potentially leveraging existing React knowledge if the team is familiar with it (as suggested by the web frontend deployment config). Expo simplifies setup, development, and build processes.
*   **UI Library:** React Native Paper (Optional, for pre-built Material Design components)
*   **Navigation:** React Navigation
*   **State Management:** React Context API or Zustand (for simplicity)
*   **API Client:** Fetch API or Axios

### 3. Core Features

*   **Prediction Input:** Allow users to input necessary data for predictions:
    *   Timestamps (e.g., using a date/time picker component)
    *   Meter IDs (e.g., text input, potentially multi-select if applicable)
    *   Context Features (e.g., structured form based on expected keys/values)
*   **Prediction Display:** Show the prediction results received from the API:
    *   Numerical predictions (potentially visualized with a simple chart)
    *   Confidence intervals
    *   Model version used for the prediction
*   **API Interaction:** Communicate with the backend `/predict` endpoint.
*   **User Feedback:** Implement loading indicators during API calls and clear error messages.

### 4. Screen Flow

1.  **Input Screen (`InputScreen.js`):**
    *   The initial screen.
    *   Contains form fields for Timestamps, Meter IDs, and Context Features.
    *   Includes validation for inputs.
    *   A "Get Prediction" button triggers the API call.
2.  **Results Screen (`ResultsScreen.js`):**
    *   Displays the prediction results fetched from the API.
    *   Shows predictions, confidence intervals, and model version.
    *   Includes a button to navigate back to the Input Screen for a new prediction.

### 5. Components

*   `PredictionForm`: Reusable component for input fields.
*   `ResultsDisplay`: Component to format and display prediction results.
*   `LoadingIndicator`: Component shown during API calls.
*   `ErrorMessage`: Component to display errors.
*   `ChartComponent` (Optional): Simple chart (e.g., line or bar) to visualize predictions if applicable.

### 6. API Integration

*   The app will make POST requests to the `/predict` endpoint of the Fluxora backend.
*   The backend API URL needs to be configurable (e.g., via environment variables or a settings screen).
*   Request Body: `PredictionRequest` schema (`timestamps`, `meter_ids`, `context_features`).
*   Response Body: `PredictionResponse` schema (`predictions`, `confidence_intervals`, `model_version`).

### 7. Styling

*   Modern, clean UI adhering to mobile design principles.
*   Responsive layout for different screen sizes.
*   Consistent theme (potentially using React Native Paper).

### 8. Deployment Strategy

*   **Primary Goal:** Mobile apps for Android & iOS.
*   **Hosting:** Since direct app store deployment is complex in this environment, we will build the application for the web using `expo build:web`.
*   **Deployment:** The resulting static web build will be deployed using the `deploy_apply_deployment` tool for permanent hosting, providing a publicly accessible URL.

### 9. Future Considerations

*   User authentication.
*   Saving prediction history.
*   Real-time data updates (if backend supports).
*   More sophisticated data visualization.

