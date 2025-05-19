from flask import Blueprint, request, jsonify
from web3 import Web3
import json
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from ..middleware.auth import token_required
from ..engine.blockchain_connector import BlockchainConnector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Blueprint
supply_chain_bp = Blueprint('supply_chain', __name__, url_prefix='/api/supply-chain')

# Initialize blockchain connector
blockchain_connector = BlockchainConnector()

# Load contract ABI
try:
    with open(os.path.join(os.path.dirname(__file__), '../contracts/SupplyChainTracker.json'), 'r') as f:
        contract_data = json.load(f)
        SUPPLY_CHAIN_ABI = contract_data['abi']
except Exception as e:
    logger.error(f"Failed to load contract ABI: {e}")
    SUPPLY_CHAIN_ABI = []

# Contract address from environment
SUPPLY_CHAIN_ADDRESS = os.getenv('SUPPLY_CHAIN_ADDRESS')

@supply_chain_bp.route('/assets', methods=['GET'])
@token_required
def get_assets():
    """Get all assets or filter by custodian"""
    try:
        custodian = request.args.get('custodian')
        
        if custodian:
            # Get assets for specific custodian
            assets = blockchain_connector.get_custodian_assets(custodian)
        else:
            # Get all assets
            asset_count = blockchain_connector.get_asset_count()
            assets = []
            for i in range(1, asset_count + 1):
                try:
                    asset = blockchain_connector.get_asset(i)
                    assets.append(asset)
                except Exception as e:
                    logger.warning(f"Error fetching asset {i}: {e}")
        
        return jsonify({"success": True, "assets": assets}), 200
    except Exception as e:
        logger.error(f"Error in get_assets: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/assets/<int:asset_id>', methods=['GET'])
@token_required
def get_asset(asset_id):
    """Get details for a specific asset"""
    try:
        asset = blockchain_connector.get_asset(asset_id)
        
        # Get transfer history
        transfers = []
        for i in range(asset['transferCount']):
            transfer = blockchain_connector.get_transfer(asset_id, i)
            transfers.append(transfer)
        
        asset['transfers'] = transfers
        
        return jsonify({"success": True, "asset": asset}), 200
    except Exception as e:
        logger.error(f"Error in get_asset: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/assets', methods=['POST'])
@token_required
def create_asset():
    """Create a new asset in the supply chain"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['metadata', 'initialCustodian', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Create asset on blockchain
        tx_hash = blockchain_connector.create_asset(
            data['metadata'],
            data['initialCustodian'],
            data['location']
        )
        
        return jsonify({
            "success": True, 
            "message": "Asset created successfully",
            "txHash": tx_hash
        }), 201
    except Exception as e:
        logger.error(f"Error in create_asset: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/assets/batch', methods=['POST'])
@token_required
def batch_create_assets():
    """Create multiple assets in a single transaction"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['metadataList', 'custodianList', 'locationList']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Validate array lengths
        if not (len(data['metadataList']) == len(data['custodianList']) == len(data['locationList'])):
            return jsonify({"success": False, "error": "Input arrays must have the same length"}), 400
        
        # Create assets on blockchain
        tx_hash = blockchain_connector.batch_create_assets(
            data['metadataList'],
            data['custodianList'],
            data['locationList']
        )
        
        return jsonify({
            "success": True, 
            "message": f"Successfully created {len(data['metadataList'])} assets",
            "txHash": tx_hash
        }), 201
    except Exception as e:
        logger.error(f"Error in batch_create_assets: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/assets/<int:asset_id>/transfer', methods=['POST'])
@token_required
def transfer_asset(asset_id):
    """Transfer an asset to a new custodian"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['to', 'location', 'proofHash']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Transfer asset on blockchain
        tx_hash = blockchain_connector.transfer_asset(
            asset_id,
            data['to'],
            data['location'],
            data['proofHash']
        )
        
        return jsonify({
            "success": True, 
            "message": "Asset transferred successfully",
            "txHash": tx_hash
        }), 200
    except Exception as e:
        logger.error(f"Error in transfer_asset: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/assets/<int:asset_id>/status', methods=['PUT'])
@token_required
def update_asset_status(asset_id):
    """Update the status of an asset"""
    try:
        data = request.json
        
        # Validate required fields
        if 'status' not in data:
            return jsonify({"success": False, "error": "Missing required field: status"}), 400
        
        # Validate status value
        valid_statuses = [0, 1, 2, 3, 4]  # Matches enum in smart contract
        if data['status'] not in valid_statuses:
            return jsonify({"success": False, "error": "Invalid status value"}), 400
        
        # Update status on blockchain
        tx_hash = blockchain_connector.update_asset_status(
            asset_id,
            data['status']
        )
        
        status_names = ["Created", "InTransit", "Delivered", "Rejected", "Recalled"]
        
        return jsonify({
            "success": True, 
            "message": f"Asset status updated to {status_names[data['status']]}",
            "txHash": tx_hash
        }), 200
    except Exception as e:
        logger.error(f"Error in update_asset_status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/assets/<int:asset_id>/location', methods=['PUT'])
@token_required
def request_location_update(asset_id):
    """Request a location update for an asset via Chainlink oracle"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['oracle', 'jobId', 'fee']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Request location update on blockchain
        tx_hash = blockchain_connector.request_location_update(
            asset_id,
            data['oracle'],
            data['jobId'],
            data['fee']
        )
        
        return jsonify({
            "success": True, 
            "message": "Location update requested successfully",
            "txHash": tx_hash
        }), 200
    except Exception as e:
        logger.error(f"Error in request_location_update: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/analytics/custodian/<address>', methods=['GET'])
@token_required
def get_custodian_analytics(address):
    """Get analytics for a specific custodian"""
    try:
        # Get assets for custodian
        assets = blockchain_connector.get_custodian_assets(address)
        
        # Calculate analytics
        total_assets = len(assets)
        status_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        for asset_id in assets:
            asset = blockchain_connector.get_asset(asset_id)
            status_counts[asset['status']] += 1
        
        status_names = ["Created", "InTransit", "Delivered", "Rejected", "Recalled"]
        status_analytics = [
            {"status": status_names[i], "count": status_counts[i]} 
            for i in range(5)
        ]
        
        return jsonify({
            "success": True,
            "analytics": {
                "totalAssets": total_assets,
                "statusBreakdown": status_analytics
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in get_custodian_analytics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@supply_chain_bp.route('/analytics/global', methods=['GET'])
@token_required
def get_global_analytics():
    """Get global supply chain analytics"""
    try:
        # Get total asset count
        asset_count = blockchain_connector.get_asset_count()
        
        # Sample a subset of assets for analytics (to avoid excessive blockchain calls)
        sample_size = min(100, asset_count)
        status_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        for i in range(1, sample_size + 1):
            try:
                asset = blockchain_connector.get_asset(i)
                status_counts[asset['status']] += 1
            except Exception as e:
                logger.warning(f"Error fetching asset {i}: {e}")
        
        # Extrapolate to full dataset if sampling
        if sample_size < asset_count:
            for status in status_counts:
                status_counts[status] = int(status_counts[status] * (asset_count / sample_size))
        
        status_names = ["Created", "InTransit", "Delivered", "Rejected", "Recalled"]
        status_analytics = [
            {"status": status_names[i], "count": status_counts[i]} 
            for i in range(5)
        ]
        
        return jsonify({
            "success": True,
            "analytics": {
                "totalAssets": asset_count,
                "statusBreakdown": status_analytics,
                "timestamp": datetime.now().isoformat()
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in get_global_analytics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
