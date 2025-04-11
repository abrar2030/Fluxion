// SPDX-License-Identifier: AGPL-3.0
pragma solidity 0.8.19;
import "@openzeppelin/contracts/utils/math/Math.sol";

contract LiquidityPoolManager {
    struct PoolConfig {
        address[] assets;
        uint256[] weights;
        uint256 fee;
        uint256 amplification;
    }
    
    mapping(bytes32 => PoolConfig) public pools;
    mapping(address => bytes32[]) public userPools;
    uint256 public constant MAX_FEE = 0.01 ether; // 1%
    
    event PoolCreated(
        bytes32 indexed poolId,
        address[] assets,
        uint256[] weights
    );

    function createPool(
        address[] calldata _assets,
        uint256[] calldata _weights,
        uint256 _fee,
        uint256 _amplification
    ) external {
        require(_fee <= MAX_FEE, "Fee too high");
        require(_assets.length == _weights.length, "Assets and weights length mismatch");
        require(_assets.length >= 2, "Minimum 2 assets required");
        
        // Generate pool ID from assets and sender
        bytes32 poolId = keccak256(abi.encodePacked(_assets, msg.sender, block.timestamp));
        
        // Store pool configuration
        pools[poolId] = PoolConfig({
            assets: _assets,
            weights: _weights,
            fee: _fee,
            amplification: _amplification
        });
        
        // Add pool to user's pools
        userPools[msg.sender].push(poolId);
        
        // Emit event
        emit PoolCreated(poolId, _assets, _weights);
    }
    
    function getPool(bytes32 _poolId) external view returns (PoolConfig memory) {
        return pools[_poolId];
    }
    
    function getUserPoolCount(address _user) external view returns (uint256) {
        return userPools[_user].length;
    }
    
    function getUserPoolAtIndex(address _user, uint256 _index) external view returns (bytes32) {
        require(_index < userPools[_user].length, "Index out of bounds");
        return userPools[_user][_index];
    }
}
