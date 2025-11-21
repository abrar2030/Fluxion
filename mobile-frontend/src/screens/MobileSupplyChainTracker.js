import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  SafeAreaView,
  StatusBar,
  FlatList,
  Image,
  Dimensions,
  Alert,
  Modal,
} from "react-native";
import { MaterialIcons, FontAwesome5, Ionicons } from "@expo/vector-icons";
import MapView, { Marker, Polyline, PROVIDER_GOOGLE } from "react-native-maps";
import QRCode from "react-native-qrcode-svg";
import { BarCodeScanner } from "expo-barcode-scanner";
import { Camera } from "expo-camera";
import * as Location from "expo-location";
import { LinearGradient } from "expo-linear-gradient";

// Get screen dimensions
const { width, height } = Dimensions.get("window");

// Mock data for demonstration
const mockAssets = [
  {
    id: 1,
    metadata: "Product XYZ-123",
    currentCustodian: "0x1234567890abcdef1234567890abcdef12345678",
    timestamp: "2023-05-18T14:30:00Z",
    status: "InTransit",
    location: "New York, USA",
    position: { latitude: 40.7128, longitude: -74.006 },
  },
  {
    id: 2,
    metadata: "Component ABC-456",
    currentCustodian: "0xabcdef1234567890abcdef1234567890abcdef12",
    timestamp: "2023-05-17T10:15:00Z",
    status: "Delivered",
    location: "Los Angeles, USA",
    position: { latitude: 34.0522, longitude: -118.2437 },
  },
  {
    id: 3,
    metadata: "Raw Material DEF-789",
    currentCustodian: "0x7890abcdef1234567890abcdef1234567890abcd",
    timestamp: "2023-05-16T08:45:00Z",
    status: "Created",
    location: "Chicago, USA",
    position: { latitude: 41.8781, longitude: -87.6298 },
  },
  {
    id: 4,
    metadata: "Shipment GHI-012",
    currentCustodian: "0xef1234567890abcdef1234567890abcdef123456",
    timestamp: "2023-05-15T16:20:00Z",
    status: "Rejected",
    location: "Houston, USA",
    position: { latitude: 29.7604, longitude: -95.3698 },
  },
];

const mockTransfers = [
  {
    assetId: 1,
    transferId: 0,
    from: "0x7890abcdef1234567890abcdef1234567890abcd",
    to: "0x1234567890abcdef1234567890abcdef12345678",
    timestamp: "2023-05-17T09:30:00Z",
    location: "Boston, USA",
    proofHash:
      "0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
  },
  {
    assetId: 2,
    transferId: 0,
    from: "0x7890abcdef1234567890abcdef1234567890abcd",
    to: "0xabcdef1234567890abcdef1234567890abcdef12",
    timestamp: "2023-05-16T11:45:00Z",
    location: "San Francisco, USA",
    proofHash:
      "0xabcdef123456789abcdef123456789abcdef123456789abcdef123456789abcd",
  },
];

// Asset route for demonstration
const assetRoute = [
  { latitude: 40.7128, longitude: -74.006 }, // New York
  { latitude: 39.9526, longitude: -75.1652 }, // Philadelphia
  { latitude: 38.9072, longitude: -77.0369 }, // Washington DC
  { latitude: 37.7749, longitude: -122.4194 }, // San Francisco
];

// Mobile Supply Chain Tracker component
const MobileSupplyChainTracker = () => {
  const [assets, setAssets] = useState([]);
  const [selectedAsset, setSelectedAsset] = useState(null);
  const [transfers, setTransfers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [showScanner, setShowScanner] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);
  const [scanned, setScanned] = useState(false);
  const [showQRCode, setShowQRCode] = useState(false);
  const [currentLocation, setCurrentLocation] = useState(null);
  const [showFilterModal, setShowFilterModal] = useState(false);
  const [showTransferModal, setShowTransferModal] = useState(false);
  const [transferDetails, setTransferDetails] = useState({
    to: "",
    location: "",
    proofHash: "",
  });

  // Fetch assets on component mount
  useEffect(() => {
    fetchAssets();
    requestLocationPermission();
  }, []);

  // Fetch asset transfers when an asset is selected
  useEffect(() => {
    if (selectedAsset) {
      fetchTransfers(selectedAsset.id);
    }
  }, [selectedAsset]);

  // Request camera permissions
  const requestCameraPermission = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    setHasPermission(status === "granted");
    if (status === "granted") {
      setShowScanner(true);
    } else {
      Alert.alert(
        "Permission Denied",
        "Please grant camera permission to scan QR codes.",
      );
    }
  };

  // Request location permissions
  const requestLocationPermission = async () => {
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status === "granted") {
      const location = await Location.getCurrentPositionAsync({});
      setCurrentLocation({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
      });
    } else {
      Alert.alert(
        "Permission Denied",
        "Please grant location permission to use all features.",
      );
    }
  };

  // Fetch assets (mock implementation)
  const fetchAssets = () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setAssets(mockAssets);
      setLoading(false);
    }, 1000);
  };

  // Fetch transfers for a specific asset (mock implementation)
  const fetchTransfers = (assetId) => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      const assetTransfers = mockTransfers.filter(
        (transfer) => transfer.assetId === assetId,
      );
      setTransfers(assetTransfers);
      setLoading(false);
    }, 800);
  };

  // Handle asset selection
  const handleAssetSelect = (asset) => {
    setSelectedAsset(asset);
  };

  // Handle search input change
  const handleSearchChange = (text) => {
    setSearchTerm(text);
  };

  // Handle status filter change
  const handleStatusFilterChange = (status) => {
    setStatusFilter(status);
    setShowFilterModal(false);
  };

  // Filter assets based on search term and status filter
  const filteredAssets = assets.filter((asset) => {
    const matchesSearch =
      asset.metadata.toLowerCase().includes(searchTerm.toLowerCase()) ||
      asset.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
      asset.id.toString().includes(searchTerm);
    const matchesStatus = statusFilter === "" || asset.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Handle barcode scan
  const handleBarCodeScanned = ({ type, data }) => {
    setScanned(true);
    setShowScanner(false);

    try {
      // Assume data is a JSON string with asset information
      const assetData = JSON.parse(data);
      Alert.alert(
        "Asset Found",
        `Found asset: ${assetData.metadata || assetData.id}`,
      );

      // Find the asset in our list
      const foundAsset = assets.find(
        (a) => a.id.toString() === assetData.id.toString(),
      );
      if (foundAsset) {
        setSelectedAsset(foundAsset);
      }
    } catch (error) {
      Alert.alert(
        "Invalid QR Code",
        "The scanned QR code is not a valid asset.",
      );
    }
  };

  // Handle transfer details form change
  const handleTransferDetailsChange = (field, value) => {
    setTransferDetails({
      ...transferDetails,
      [field]: value,
    });
  };

  // Transfer asset (mock implementation)
  const handleTransferAsset = () => {
    if (!selectedAsset) return;

    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      const newTransferId = transfers.length;
      const newTransfer = {
        assetId: selectedAsset.id,
        transferId: newTransferId,
        from: selectedAsset.currentCustodian,
        to: transferDetails.to,
        timestamp: new Date().toISOString(),
        location:
          transferDetails.location ||
          (currentLocation ? "Current Location" : "Unknown"),
        proofHash:
          transferDetails.proofHash ||
          "0x" +
            Array(64)
              .fill(0)
              .map(() => Math.floor(Math.random() * 16).toString(16))
              .join(""),
      };

      // Update asset
      const updatedAssets = assets.map((asset) => {
        if (asset.id === selectedAsset.id) {
          return {
            ...asset,
            currentCustodian: transferDetails.to,
            location:
              transferDetails.location ||
              (currentLocation ? "Current Location" : "Unknown"),
            status: "InTransit",
            timestamp: new Date().toISOString(),
          };
        }
        return asset;
      });

      setAssets(updatedAssets);
      setTransfers([...transfers, newTransfer]);
      setSelectedAsset({
        ...selectedAsset,
        currentCustodian: transferDetails.to,
        location:
          transferDetails.location ||
          (currentLocation ? "Current Location" : "Unknown"),
        status: "InTransit",
        timestamp: new Date().toISOString(),
      });

      setLoading(false);
      setShowTransferModal(false);
      setTransferDetails({
        to: "",
        location: "",
        proofHash: "",
      });

      Alert.alert(
        "Success",
        `Asset ${selectedAsset.id} transferred successfully`,
      );
    }, 1500);
  };

  // Format address for display
  const formatAddress = (address) => {
    if (!address) return "";
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case "Created":
        return "#3498db";
      case "InTransit":
        return "#f39c12";
      case "Delivered":
        return "#2ecc71";
      case "Rejected":
        return "#e74c3c";
      case "Recalled":
        return "#9b59b6";
      default:
        return "#7f8c8d";
    }
  };

  // Render asset item
  const renderAssetItem = ({ item }) => (
    <TouchableOpacity
      style={[
        styles.assetItem,
        selectedAsset?.id === item.id && styles.selectedAssetItem,
      ]}
      onPress={() => handleAssetSelect(item)}
    >
      <View style={styles.assetHeader}>
        <Text style={styles.assetTitle}>Asset #{item.id}</Text>
        <View
          style={[
            styles.statusBadge,
            { backgroundColor: getStatusColor(item.status) + "20" },
          ]}
        >
          <Text
            style={[styles.statusText, { color: getStatusColor(item.status) }]}
          >
            {item.status}
          </Text>
        </View>
      </View>
      <Text style={styles.assetMetadata}>{item.metadata}</Text>
      <View style={styles.assetDetail}>
        <MaterialIcons name="location-on" size={16} color="#7f8c8d" />
        <Text style={styles.assetDetailText}>{item.location}</Text>
      </View>
      <View style={styles.assetDetail}>
        <MaterialIcons name="access-time" size={16} color="#7f8c8d" />
        <Text style={styles.assetDetailText}>
          {new Date(item.timestamp).toLocaleString()}
        </Text>
      </View>
    </TouchableOpacity>
  );

  // Render transfer item
  const renderTransferItem = ({ item }) => (
    <View style={styles.transferItem}>
      <View style={styles.transferHeader}>
        <Text style={styles.transferTitle}>Transfer #{item.transferId}</Text>
        <Text style={styles.transferDate}>
          {new Date(item.timestamp).toLocaleString()}
        </Text>
      </View>
      <View style={styles.transferDetail}>
        <Text style={styles.transferLabel}>From:</Text>
        <Text style={styles.transferValue}>{formatAddress(item.from)}</Text>
      </View>
      <View style={styles.transferDetail}>
        <Text style={styles.transferLabel}>To:</Text>
        <Text style={styles.transferValue}>{formatAddress(item.to)}</Text>
      </View>
      <View style={styles.transferDetail}>
        <Text style={styles.transferLabel}>Location:</Text>
        <Text style={styles.transferValue}>{item.location}</Text>
      </View>
      <View style={styles.transferDetail}>
        <Text style={styles.transferLabel}>Proof Hash:</Text>
        <Text style={styles.transferHash}>{item.proofHash}</Text>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f8f9fa" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Supply Chain Tracker</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity
            style={styles.iconButton}
            onPress={() => setShowFilterModal(true)}
          >
            <MaterialIcons name="filter-list" size={24} color="#333" />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.iconButton}
            onPress={requestCameraPermission}
          >
            <MaterialIcons name="qr-code-scanner" size={24} color="#333" />
          </TouchableOpacity>
        </View>
      </View>

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <MaterialIcons
          name="search"
          size={20}
          color="#7f8c8d"
          style={styles.searchIcon}
        />
        <TextInput
          style={styles.searchInput}
          placeholder="Search assets..."
          value={searchTerm}
          onChangeText={handleSearchChange}
        />
        {searchTerm !== "" && (
          <TouchableOpacity onPress={() => setSearchTerm("")}>
            <MaterialIcons name="clear" size={20} color="#7f8c8d" />
          </TouchableOpacity>
        )}
      </View>

      {/* Status Filter Chips */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.filterChips}
      >
        <TouchableOpacity
          style={[
            styles.filterChip,
            statusFilter === "" && styles.activeFilterChip,
          ]}
          onPress={() => handleStatusFilterChange("")}
        >
          <Text
            style={[
              styles.filterChipText,
              statusFilter === "" && styles.activeFilterChipText,
            ]}
          >
            All
          </Text>
        </TouchableOpacity>
        {["Created", "InTransit", "Delivered", "Rejected", "Recalled"].map(
          (status) => (
            <TouchableOpacity
              key={status}
              style={[
                styles.filterChip,
                statusFilter === status && styles.activeFilterChip,
                statusFilter === status && {
                  backgroundColor: getStatusColor(status) + "20",
                },
              ]}
              onPress={() => handleStatusFilterChange(status)}
            >
              <Text
                style={[
                  styles.filterChipText,
                  statusFilter === status && styles.activeFilterChipText,
                  statusFilter === status && { color: getStatusColor(status) },
                ]}
              >
                {status}
              </Text>
            </TouchableOpacity>
          ),
        )}
      </ScrollView>

      {/* Content */}
      {loading && assets.length === 0 ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#3498db" />
          <Text style={styles.loadingText}>Loading assets...</Text>
        </View>
      ) : selectedAsset ? (
        <ScrollView style={styles.assetDetailContainer}>
          {/* Asset Details Header */}
          <View style={styles.assetDetailHeader}>
            <View style={styles.assetDetailHeaderContent}>
              <Text style={styles.assetDetailTitle}>
                Asset #{selectedAsset.id}
              </Text>
              <View
                style={[
                  styles.statusBadge,
                  {
                    backgroundColor:
                      getStatusColor(selectedAsset.status) + "20",
                  },
                ]}
              >
                <Text
                  style={[
                    styles.statusText,
                    { color: getStatusColor(selectedAsset.status) },
                  ]}
                >
                  {selectedAsset.status}
                </Text>
              </View>
            </View>
            <TouchableOpacity
              style={styles.transferButton}
              onPress={() => setShowTransferModal(true)}
            >
              <Text style={styles.transferButtonText}>Transfer</Text>
              <MaterialIcons name="arrow-forward" size={16} color="#fff" />
            </TouchableOpacity>
          </View>

          {/* Asset Details */}
          <View style={styles.detailCard}>
            <Text style={styles.detailCardTitle}>Asset Information</Text>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Metadata:</Text>
              <Text style={styles.detailValue}>{selectedAsset.metadata}</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Current Custodian:</Text>
              <Text style={styles.detailValue}>
                {formatAddress(selectedAsset.currentCustodian)}
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Location:</Text>
              <Text style={styles.detailValue}>{selectedAsset.location}</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Last Updated:</Text>
              <Text style={styles.detailValue}>
                {new Date(selectedAsset.timestamp).toLocaleString()}
              </Text>
            </View>
          </View>

          {/* Asset QR Code */}
          <TouchableOpacity
            style={styles.qrCodeCard}
            onPress={() => setShowQRCode(true)}
          >
            <View style={styles.qrCodePreview}>
              <QRCode
                value={JSON.stringify({
                  id: selectedAsset.id,
                  metadata: selectedAsset.metadata,
                })}
                size={80}
              />
            </View>
            <View style={styles.qrCodeInfo}>
              <Text style={styles.qrCodeTitle}>Asset QR Code</Text>
              <Text style={styles.qrCodeSubtitle}>Tap to view full size</Text>
            </View>
            <MaterialIcons name="chevron-right" size={24} color="#7f8c8d" />
          </TouchableOpacity>

          {/* Asset Map */}
          <View style={styles.mapCard}>
            <Text style={styles.mapCardTitle}>Asset Location</Text>
            <View style={styles.mapContainer}>
              <MapView
                provider={PROVIDER_GOOGLE}
                style={styles.map}
                initialRegion={{
                  latitude: selectedAsset.position?.latitude || 40.7128,
                  longitude: selectedAsset.position?.longitude || -74.006,
                  latitudeDelta: 0.0922,
                  longitudeDelta: 0.0421,
                }}
              >
                <Marker
                  coordinate={
                    selectedAsset.position || {
                      latitude: 40.7128,
                      longitude: -74.006,
                    }
                  }
                  title={`Asset #${selectedAsset.id}`}
                  description={selectedAsset.metadata}
                />
                <Polyline
                  coordinates={assetRoute}
                  strokeColor="#3498db"
                  strokeWidth={3}
                />
              </MapView>
            </View>
          </View>

          {/* Transfer History */}
          <View style={styles.transfersCard}>
            <Text style={styles.transfersCardTitle}>Transfer History</Text>
            {loading ? (
              <ActivityIndicator
                size="small"
                color="#3498db"
                style={{ marginVertical: 20 }}
              />
            ) : transfers.length === 0 ? (
              <Text style={styles.noTransfersText}>
                No transfer history available for this asset.
              </Text>
            ) : (
              <FlatList
                data={transfers}
                renderItem={renderTransferItem}
                keyExtractor={(item) => `${item.assetId}-${item.transferId}`}
                scrollEnabled={false}
              />
            )}
          </View>
        </ScrollView>
      ) : filteredAssets.length === 0 ? (
        <View style={styles.emptyContainer}>
          <MaterialIcons name="search-off" size={64} color="#bdc3c7" />
          <Text style={styles.emptyText}>
            No assets found matching your criteria
          </Text>
        </View>
      ) : (
        <FlatList
          data={filteredAssets}
          renderItem={renderAssetItem}
          keyExtractor={(item) => item.id.toString()}
          contentContainerStyle={styles.assetList}
        />
      )}

      {/* QR Code Scanner Modal */}
      <Modal
        visible={showScanner}
        animationType="slide"
        onRequestClose={() => setShowScanner(false)}
      >
        <SafeAreaView style={styles.scannerContainer}>
          <View style={styles.scannerHeader}>
            <TouchableOpacity onPress={() => setShowScanner(false)}>
              <MaterialIcons name="close" size={24} color="#fff" />
            </TouchableOpacity>
            <Text style={styles.scannerTitle}>Scan Asset QR Code</Text>
            <View style={{ width: 24 }} />
          </View>

          {hasPermission === null ? (
            <View style={styles.scannerPlaceholder}>
              <Text>Requesting camera permission...</Text>
            </View>
          ) : hasPermission === false ? (
            <View style={styles.scannerPlaceholder}>
              <Text>No access to camera</Text>
            </View>
          ) : (
            <View style={styles.cameraContainer}>
              <BarCodeScanner
                onBarCodeScanned={scanned ? undefined : handleBarCodeScanned}
                style={StyleSheet.absoluteFillObject}
              />
              <View style={styles.scannerOverlay}>
                <View style={styles.scannerTargetCorner} />
                <View
                  style={[
                    styles.scannerTargetCorner,
                    { top: 0, right: 0, transform: [{ rotate: "90deg" }] },
                  ]}
                />
                <View
                  style={[
                    styles.scannerTargetCorner,
                    { bottom: 0, right: 0, transform: [{ rotate: "180deg" }] },
                  ]}
                />
                <View
                  style={[
                    styles.scannerTargetCorner,
                    { bottom: 0, left: 0, transform: [{ rotate: "270deg" }] },
                  ]}
                />
              </View>
              <View style={styles.scannerInstructions}>
                <Text style={styles.scannerInstructionsText}>
                  Position QR code within the frame
                </Text>
              </View>
            </View>
          )}

          {scanned && (
            <TouchableOpacity
              style={styles.rescanButton}
              onPress={() => setScanned(false)}
            >
              <Text style={styles.rescanButtonText}>Tap to Scan Again</Text>
            </TouchableOpacity>
          )}
        </SafeAreaView>
      </Modal>

      {/* QR Code Full Screen Modal */}
      <Modal
        visible={showQRCode}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setShowQRCode(false)}
      >
        <View style={styles.qrCodeModalContainer}>
          <View style={styles.qrCodeModalContent}>
            <View style={styles.qrCodeModalHeader}>
              <Text style={styles.qrCodeModalTitle}>
                Asset #{selectedAsset?.id} QR Code
              </Text>
              <TouchableOpacity onPress={() => setShowQRCode(false)}>
                <MaterialIcons name="close" size={24} color="#333" />
              </TouchableOpacity>
            </View>
            <View style={styles.qrCodeModalBody}>
              {selectedAsset && (
                <QRCode
                  value={JSON.stringify({
                    id: selectedAsset.id,
                    metadata: selectedAsset.metadata,
                  })}
                  size={250}
                />
              )}
            </View>
            <Text style={styles.qrCodeModalSubtitle}>
              Scan this code to quickly access asset information
            </Text>
          </View>
        </View>
      </Modal>

      {/* Filter Modal */}
      <Modal
        visible={showFilterModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowFilterModal(false)}
      >
        <View style={styles.filterModalContainer}>
          <View style={styles.filterModalContent}>
            <View style={styles.filterModalHeader}>
              <Text style={styles.filterModalTitle}>Filter Assets</Text>
              <TouchableOpacity onPress={() => setShowFilterModal(false)}>
                <MaterialIcons name="close" size={24} color="#333" />
              </TouchableOpacity>
            </View>
            <View style={styles.filterModalBody}>
              <Text style={styles.filterSectionTitle}>Status</Text>
              <TouchableOpacity
                style={[
                  styles.filterOption,
                  statusFilter === "" && styles.filterOptionSelected,
                ]}
                onPress={() => handleStatusFilterChange("")}
              >
                <Text style={styles.filterOptionText}>All</Text>
                {statusFilter === "" && (
                  <MaterialIcons name="check" size={20} color="#3498db" />
                )}
              </TouchableOpacity>
              {[
                "Created",
                "InTransit",
                "Delivered",
                "Rejected",
                "Recalled",
              ].map((status) => (
                <TouchableOpacity
                  key={status}
                  style={[
                    styles.filterOption,
                    statusFilter === status && styles.filterOptionSelected,
                  ]}
                  onPress={() => handleStatusFilterChange(status)}
                >
                  <View style={styles.filterOptionContent}>
                    <View
                      style={[
                        styles.statusDot,
                        { backgroundColor: getStatusColor(status) },
                      ]}
                    />
                    <Text style={styles.filterOptionText}>{status}</Text>
                  </View>
                  {statusFilter === status && (
                    <MaterialIcons name="check" size={20} color="#3498db" />
                  )}
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </View>
      </Modal>

      {/* Transfer Modal */}
      <Modal
        visible={showTransferModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowTransferModal(false)}
      >
        <View style={styles.transferModalContainer}>
          <View style={styles.transferModalContent}>
            <View style={styles.transferModalHeader}>
              <Text style={styles.transferModalTitle}>
                Transfer Asset #{selectedAsset?.id}
              </Text>
              <TouchableOpacity onPress={() => setShowTransferModal(false)}>
                <MaterialIcons name="close" size={24} color="#333" />
              </TouchableOpacity>
            </View>
            <View style={styles.transferModalBody}>
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Recipient Address</Text>
                <TextInput
                  style={styles.input}
                  placeholder="0x..."
                  value={transferDetails.to}
                  onChangeText={(text) =>
                    handleTransferDetailsChange("to", text)
                  }
                />
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>New Location</Text>
                <View style={styles.locationInputContainer}>
                  <TextInput
                    style={styles.locationInput}
                    placeholder="City, Country"
                    value={transferDetails.location}
                    onChangeText={(text) =>
                      handleTransferDetailsChange("location", text)
                    }
                  />
                  {currentLocation && (
                    <TouchableOpacity
                      style={styles.useLocationButton}
                      onPress={() =>
                        handleTransferDetailsChange(
                          "location",
                          "Current Location",
                        )
                      }
                    >
                      <MaterialIcons
                        name="my-location"
                        size={20}
                        color="#3498db"
                      />
                    </TouchableOpacity>
                  )}
                </View>
              </View>

              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Proof Hash (Optional)</Text>
                <TextInput
                  style={styles.input}
                  placeholder="0x..."
                  value={transferDetails.proofHash}
                  onChangeText={(text) =>
                    handleTransferDetailsChange("proofHash", text)
                  }
                />
                <Text style={styles.inputHelp}>
                  Leave blank to generate automatically
                </Text>
              </View>
            </View>
            <View style={styles.transferModalFooter}>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => setShowTransferModal(false)}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.confirmButton,
                  (!transferDetails.to || loading) && styles.disabledButton,
                ]}
                onPress={handleTransferAsset}
                disabled={!transferDetails.to || loading}
              >
                {loading ? (
                  <ActivityIndicator size="small" color="#fff" />
                ) : (
                  <Text style={styles.confirmButtonText}>Transfer Asset</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Back Button (when viewing asset details) */}
      {selectedAsset && (
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => setSelectedAsset(null)}
        >
          <MaterialIcons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8f9fa",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#e0e0e0",
    backgroundColor: "#fff",
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#333",
  },
  headerActions: {
    flexDirection: "row",
    alignItems: "center",
  },
  iconButton: {
    padding: 8,
    marginLeft: 8,
  },
  searchContainer: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#fff",
    borderRadius: 8,
    margin: 16,
    paddingHorizontal: 12,
    paddingVertical: 8,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    height: 40,
    fontSize: 16,
  },
  filterChips: {
    paddingHorizontal: 12,
    marginBottom: 12,
  },
  filterChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    backgroundColor: "#f0f0f0",
    marginRight: 8,
  },
  activeFilterChip: {
    backgroundColor: "#3498db20",
  },
  filterChipText: {
    fontSize: 14,
    color: "#7f8c8d",
  },
  activeFilterChipText: {
    color: "#3498db",
    fontWeight: "500",
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: "#7f8c8d",
  },
  emptyContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  emptyText: {
    marginTop: 16,
    fontSize: 16,
    color: "#7f8c8d",
    textAlign: "center",
  },
  assetList: {
    padding: 16,
  },
  assetItem: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  selectedAssetItem: {
    borderLeftWidth: 4,
    borderLeftColor: "#3498db",
  },
  assetHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  assetTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusText: {
    fontSize: 12,
    fontWeight: "500",
  },
  assetMetadata: {
    fontSize: 14,
    color: "#333",
    marginBottom: 8,
  },
  assetDetail: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 4,
  },
  assetDetailText: {
    fontSize: 12,
    color: "#7f8c8d",
    marginLeft: 4,
  },
  assetDetailContainer: {
    flex: 1,
  },
  assetDetailHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#e0e0e0",
  },
  assetDetailHeaderContent: {
    flex: 1,
  },
  assetDetailTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 4,
  },
  transferButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#3498db",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 4,
  },
  transferButtonText: {
    color: "#fff",
    fontWeight: "500",
    marginRight: 4,
  },
  detailCard: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 16,
    margin: 16,
    marginBottom: 8,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  detailCardTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: "row",
    marginBottom: 8,
  },
  detailLabel: {
    width: 120,
    fontSize: 14,
    color: "#7f8c8d",
  },
  detailValue: {
    flex: 1,
    fontSize: 14,
    color: "#333",
  },
  qrCodeCard: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 16,
    margin: 16,
    marginTop: 8,
    marginBottom: 8,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    flexDirection: "row",
    alignItems: "center",
  },
  qrCodePreview: {
    padding: 8,
    backgroundColor: "#f8f9fa",
    borderRadius: 4,
  },
  qrCodeInfo: {
    flex: 1,
    marginLeft: 16,
  },
  qrCodeTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
  },
  qrCodeSubtitle: {
    fontSize: 12,
    color: "#7f8c8d",
    marginTop: 4,
  },
  mapCard: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 16,
    margin: 16,
    marginTop: 8,
    marginBottom: 8,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  mapCardTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 12,
  },
  mapContainer: {
    height: 200,
    borderRadius: 8,
    overflow: "hidden",
  },
  map: {
    ...StyleSheet.absoluteFillObject,
  },
  transfersCard: {
    backgroundColor: "#fff",
    borderRadius: 8,
    padding: 16,
    margin: 16,
    marginTop: 8,
    marginBottom: 24,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  transfersCardTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 12,
  },
  noTransfersText: {
    fontSize: 14,
    color: "#7f8c8d",
    fontStyle: "italic",
    textAlign: "center",
    marginVertical: 16,
  },
  transferItem: {
    borderWidth: 1,
    borderColor: "#e0e0e0",
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  transferHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  transferTitle: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#333",
  },
  transferDate: {
    fontSize: 12,
    color: "#7f8c8d",
  },
  transferDetail: {
    flexDirection: "row",
    marginBottom: 4,
  },
  transferLabel: {
    width: 80,
    fontSize: 14,
    color: "#7f8c8d",
  },
  transferValue: {
    flex: 1,
    fontSize: 14,
    color: "#333",
  },
  transferHash: {
    flex: 1,
    fontSize: 14,
    color: "#333",
    fontFamily: "monospace",
  },
  backButton: {
    position: "absolute",
    bottom: 24,
    left: 24,
    backgroundColor: "#3498db",
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: "center",
    alignItems: "center",
    elevation: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  scannerContainer: {
    flex: 1,
    backgroundColor: "#000",
  },
  scannerHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
  },
  scannerTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#fff",
  },
  scannerPlaceholder: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  cameraContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  scannerOverlay: {
    width: 250,
    height: 250,
    borderWidth: 2,
    borderColor: "rgba(255, 255, 255, 0.5)",
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
  },
  scannerTargetCorner: {
    position: "absolute",
    top: 0,
    left: 0,
    width: 30,
    height: 30,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderColor: "#3498db",
  },
  scannerInstructions: {
    position: "absolute",
    bottom: 100,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  scannerInstructionsText: {
    color: "#fff",
    fontSize: 16,
    textAlign: "center",
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  rescanButton: {
    backgroundColor: "#3498db",
    paddingVertical: 12,
    alignItems: "center",
    margin: 16,
    borderRadius: 8,
  },
  rescanButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "bold",
  },
  qrCodeModalContainer: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  qrCodeModalContent: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 24,
    width: "80%",
    alignItems: "center",
  },
  qrCodeModalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    width: "100%",
    marginBottom: 24,
  },
  qrCodeModalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#333",
  },
  qrCodeModalBody: {
    padding: 16,
    backgroundColor: "#f8f9fa",
    borderRadius: 8,
    marginBottom: 16,
  },
  qrCodeModalSubtitle: {
    fontSize: 14,
    color: "#7f8c8d",
    textAlign: "center",
  },
  filterModalContainer: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "flex-end",
  },
  filterModalContent: {
    backgroundColor: "#fff",
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 24,
  },
  filterModalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 24,
  },
  filterModalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#333",
  },
  filterModalBody: {
    marginBottom: 24,
  },
  filterSectionTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#333",
    marginBottom: 12,
  },
  filterOption: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#f0f0f0",
  },
  filterOptionSelected: {
    backgroundColor: "#f8f9fa",
  },
  filterOptionContent: {
    flexDirection: "row",
    alignItems: "center",
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  filterOptionText: {
    fontSize: 16,
    color: "#333",
  },
  transferModalContainer: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    justifyContent: "flex-end",
  },
  transferModalContent: {
    backgroundColor: "#fff",
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 24,
  },
  transferModalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 24,
  },
  transferModalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#333",
  },
  transferModalBody: {
    marginBottom: 24,
  },
  inputGroup: {
    marginBottom: 16,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: "500",
    color: "#333",
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: "#e0e0e0",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  inputHelp: {
    fontSize: 12,
    color: "#7f8c8d",
    marginTop: 4,
  },
  locationInputContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  locationInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#e0e0e0",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  useLocationButton: {
    padding: 10,
    marginLeft: 8,
  },
  transferModalFooter: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  cancelButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e0e0e0",
    borderRadius: 8,
    marginRight: 8,
  },
  cancelButtonText: {
    fontSize: 16,
    color: "#333",
  },
  confirmButton: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
    backgroundColor: "#3498db",
    borderRadius: 8,
    marginLeft: 8,
  },
  confirmButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#fff",
  },
  disabledButton: {
    backgroundColor: "#bdc3c7",
  },
});

export default MobileSupplyChainTracker;
