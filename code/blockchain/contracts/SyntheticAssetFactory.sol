// SPDX-License-Identifier: AGPL-3.0
pragma solidity 0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@chainlink/contracts/src/v0.8/ChainlinkClient.sol";

contract SyntheticAssetFactory is Ownable, ChainlinkClient {
    using Chainlink for Chainlink.Request;
    
    struct SyntheticAsset {
        address token;
        address oracle;
        bytes32 jobId;
        uint256 fee;
        bool active;
    }
    
    mapping(bytes32 => SyntheticAsset) public syntheticAssets;
    bytes32[] public assetIds;
    
    event SyntheticAssetCreated(
        bytes32 indexed assetId,
        address token,
        address oracle,
        bytes32 jobId
    );
    
    constructor() Ownable() {
        setPublicChainlinkToken();
    }
    
    function createSynthetic(
        bytes32 _assetId,
        address _oracle,
        bytes32 _jobId,
        uint256 _fee
    ) external onlyOwner {
        require(syntheticAssets[_assetId].token == address(0), "Asset already exists");
        
        // Deploy new ERC20 token for the synthetic asset
        SyntheticToken token = new SyntheticToken(
            string(abi.encodePacked("Synthetic ", bytes32ToString(_assetId))),
            string(abi.encodePacked("s", bytes32ToString(_assetId)))
        );
        
        // Store synthetic asset configuration
        syntheticAssets[_assetId] = SyntheticAsset({
            token: address(token),
            oracle: _oracle,
            jobId: _jobId,
            fee: _fee,
            active: true
        });
        
        // Add asset ID to list
        assetIds.push(_assetId);
        
        // Emit event
        emit SyntheticAssetCreated(_assetId, address(token), _oracle, _jobId);
    }
    
    function requestPriceUpdate(bytes32 _assetId) external {
        SyntheticAsset memory asset = syntheticAssets[_assetId];
        require(asset.active, "Asset not active");
        
        Chainlink.Request memory req = buildChainlinkRequest(
            asset.jobId,
            address(this),
            this.fulfillPriceUpdate.selector
        );
        
        // Add asset ID as parameter
        req.add("assetId", bytes32ToString(_assetId));
        
        // Send request
        sendChainlinkRequestTo(asset.oracle, req, asset.fee);
    }
    
    function fulfillPriceUpdate(bytes32 _requestId, uint256 _price) external recordChainlinkFulfillment(_requestId) {
        // Implementation for price update logic
        // This would update the price of the synthetic asset
    }
    
    function deactivateAsset(bytes32 _assetId) external onlyOwner {
        require(syntheticAssets[_assetId].token != address(0), "Asset does not exist");
        syntheticAssets[_assetId].active = false;
    }
    
    function reactivateAsset(bytes32 _assetId) external onlyOwner {
        require(syntheticAssets[_assetId].token != address(0), "Asset does not exist");
        syntheticAssets[_assetId].active = true;
    }
    
    function getAssetCount() external view returns (uint256) {
        return assetIds.length;
    }
    
    // Helper function to convert bytes32 to string
    function bytes32ToString(bytes32 _bytes32) internal pure returns (string memory) {
        uint8 i = 0;
        while(i < 32 && _bytes32[i] != 0) {
            i++;
        }
        bytes memory bytesArray = new bytes(i);
        for (i = 0; i < 32 && _bytes32[i] != 0; i++) {
            bytesArray[i] = _bytes32[i];
        }
        return string(bytesArray);
    }
}

// Synthetic token contract
contract SyntheticToken is ERC20Burnable, Ownable {
    constructor(string memory name, string memory symbol) ERC20(name, symbol) Ownable() {}
    
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }
}
