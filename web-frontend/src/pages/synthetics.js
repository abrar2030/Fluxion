import { ethers } from 'ethers';

// A more complete, albeit still example, ABI for a synthetic asset factory
export const SYNTHETIC_ABI = [
    // Event emitted when a new synthetic asset is created
    'event SyntheticAssetCreated(address indexed assetAddress, string assetId, address oracle, bytes32 jobId, uint250 paymentAmount)',

    // Function to create a new synthetic asset
    'function createSynthetic(string memory assetId, address oracle, bytes32 jobId, uint256 paymentAmount) external returns (address)',

    // Function to get the address of a synthetic asset by its ID
    'function getSyntheticAsset(string memory assetId) external view returns (address)',

    // Function to check if a synthetic asset ID exists
    'function syntheticAssetExists(string memory assetId) external view returns (bool)',

    // Owner-only functions (examples)
    'function setFee(uint256 newFee) external', // Example: Set a creation fee
    'function withdrawFees() external', // Example: Withdraw collected fees
    'function owner() external view returns (address)',
    'function transferOwnership(address newOwner) external',
];

export async function createSyntheticAsset(provider, params) {
    if (!provider || !params || !process.env.FACTORY_ADDRESS) {
        console.error('Missing provider, params, or FACTORY_ADDRESS for createSyntheticAsset');
        throw new Error('Configuration error for creating synthetic asset.');
    }
    const signer = provider.getSigner();
    const factory = new ethers.Contract(process.env.FACTORY_ADDRESS, SYNTHETIC_ABI, signer);

    try {
        const tx = await factory.createSynthetic(
            params.assetId, // e.g., "sUSD"
            params.oracle, // e.g., Chainlink oracle address for the underlying asset price
            params.jobId, // e.g., Chainlink Job ID for the oracle
            ethers.utils.parseEther(params.payment.toString()), // e.g., Collateral amount
        );

        return tx.wait(); // Returns a promise that resolves to the transaction receipt
    } catch (error) {
        console.error('Error creating synthetic asset:', error);
        throw error; // Re-throw the error to be handled by the caller
    }
}
