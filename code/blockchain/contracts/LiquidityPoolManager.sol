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
    uint256 public constant MAX_FEE = 0.01 ether; // 1%

    function createPool(
        address[] calldata _assets,
        uint256[] calldata _weights,
        uint256 _fee,
        uint256 _amplification
    ) external {
        require(_fee <= MAX_FEE, "Fee too high");
        require(_assets.length == _weights.length,