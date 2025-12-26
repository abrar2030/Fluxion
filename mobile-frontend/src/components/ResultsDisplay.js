import React from 'react';
import { StyleSheet, ScrollView, View } from 'react-native';
import { Card, Text, Title, Paragraph, useTheme } from 'react-native-paper';

const ResultsDisplay = ({ results }) => {
    const theme = useTheme();

    if (!results) {
        return (
            <Card style={styles.card}>
                <Card.Content>
                    <Paragraph>No prediction results available.</Paragraph>
                </Card.Content>
            </Card>
        );
    }

    const { predictions, confidence_intervals, model_version } = results;

    // Check if predictions is an object (like the mock data)
    const isObject =
        typeof predictions === 'object' && predictions !== null && !Array.isArray(predictions);
    const meterIds = isObject ? Object.keys(predictions) : [];

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
            <Card style={styles.card}>
                <Card.Content>
                    <Title style={styles.title}>Prediction Results</Title>
                    <Paragraph
                        style={[styles.modelVersion, { color: theme.colors.onSurfaceVariant }]}
                    >
                        Model Version: {model_version || 'N/A'}
                    </Paragraph>
                </Card.Content>
            </Card>

            {isObject ? (
                meterIds.map((meterId) => (
                    <Card key={meterId} style={styles.card}>
                        <Card.Title title={`Meter ID: ${meterId}`} titleStyle={styles.meterTitle} />
                        <Card.Content>
                            {predictions[meterId].map((prediction, index) => (
                                <View key={index} style={styles.predictionItem}>
                                    <Text style={styles.predictionLabel}>
                                        Prediction {index + 1}:
                                    </Text>
                                    <Text
                                        style={[
                                            styles.predictionValue,
                                            { color: theme.colors.primary },
                                        ]}
                                    >
                                        {prediction.toFixed(4)}
                                    </Text>
                                    {confidence_intervals &&
                                        confidence_intervals[meterId] &&
                                        confidence_intervals[meterId][index] && (
                                            <Text
                                                style={[
                                                    styles.confidenceInterval,
                                                    { color: theme.colors.onSurfaceVariant },
                                                ]}
                                            >
                                                95% CI: [
                                                {confidence_intervals[meterId][index][0].toFixed(4)}
                                                ,{' '}
                                                {confidence_intervals[meterId][index][1].toFixed(4)}
                                                ]
                                            </Text>
                                        )}
                                </View>
                            ))}
                        </Card.Content>
                    </Card>
                ))
            ) : (
                // Fallback for array structure (original code's assumption)
                <Card style={styles.card}>
                    <Card.Content>
                        {predictions.map((prediction, index) => (
                            <View key={index} style={styles.predictionItem}>
                                <Text style={styles.predictionLabel}>Prediction {index + 1}:</Text>
                                <Text
                                    style={[
                                        styles.predictionValue,
                                        { color: theme.colors.primary },
                                    ]}
                                >
                                    {prediction.toFixed(4)}
                                </Text>
                                {confidence_intervals && confidence_intervals[index] && (
                                    <Text
                                        style={[
                                            styles.confidenceInterval,
                                            { color: theme.colors.onSurfaceVariant },
                                        ]}
                                    >
                                        95% CI: [{confidence_intervals[index][0].toFixed(4)},{' '}
                                        {confidence_intervals[index][1].toFixed(4)}]
                                    </Text>
                                )}
                            </View>
                        ))}
                    </Card.Content>
                </Card>
            )}
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
    title: {
        textAlign: 'center',
        marginBottom: 8,
    },
    modelVersion: {
        textAlign: 'center',
        marginBottom: 8,
        fontSize: 14,
    },
    meterTitle: {
        fontWeight: 'bold',
    },
    predictionItem: {
        marginBottom: 12,
        paddingBottom: 8,
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    predictionLabel: {
        fontSize: 16,
        fontWeight: 'bold',
        marginBottom: 4,
    },
    predictionValue: {
        fontSize: 18,
        marginBottom: 4,
    },
    confidenceInterval: {
        fontSize: 14,
    },
});

export default ResultsDisplay;
