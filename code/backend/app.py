from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import redis
import json
import logging
from datetime import datetime, timedelta

# Import blueprints
from api.supply_chain_routes import supply_chain_bp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fluxion_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Redis for caching
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', '')
redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    decode_responses=True
)

# Test Redis connection
try:
    redis_client.ping()
    logger.info("Redis connection successful")
except redis.ConnectionError:
    logger.warning("Redis connection failed, caching will be disabled")
    redis_client = None

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(supply_chain_bp)

# Cache middleware
def get_cached_response(key, expiry=300):
    """Get cached response from Redis"""
    if redis_client:
        cached = redis_client.get(key)
        if cached:
            logger.info(f"Cache hit for {key}")
            return json.loads(cached)
    return None

def set_cached_response(key, data, expiry=300):
    """Set cached response in Redis"""
    if redis_client:
        redis_client.setex(key, expiry, json.dumps(data))
        logger.info(f"Cache set for {key}, expires in {expiry}s")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.path}")
    return jsonify({"error": "Resource not found", "path": request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return jsonify({"error": "Internal server error", "details": str(error)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "redis": "connected" if redis_client else "disconnected",
            "blockchain": "connected"  # This should be dynamically checked
        }
    }
    return jsonify(status), 200

# Main route
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to Fluxion API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": [
            "/api/supply-chain/assets",
            "/api/supply-chain/analytics/global",
            "/health"
        ]
    })

# Documentation route
@app.route('/docs', methods=['GET'])
def docs():
    return jsonify({
        "title": "Fluxion API Documentation",
        "version": "2.0.0",
        "description": "API for interacting with the Fluxion blockchain-based supply chain system",
        "endpoints": {
            "supply_chain": {
                "base_url": "/api/supply-chain",
                "endpoints": [
                    {
                        "path": "/assets",
                        "method": "GET",
                        "description": "Get all assets or filter by custodian",
                        "params": [
                            {"name": "custodian", "type": "string", "required": False}
                        ]
                    },
                    {
                        "path": "/assets/{asset_id}",
                        "method": "GET",
                        "description": "Get details for a specific asset"
                    },
                    {
                        "path": "/assets",
                        "method": "POST",
                        "description": "Create a new asset in the supply chain",
                        "body": {
                            "metadata": "string",
                            "initialCustodian": "address",
                            "location": "string"
                        }
                    },
                    {
                        "path": "/assets/batch",
                        "method": "POST",
                        "description": "Create multiple assets in a single transaction",
                        "body": {
                            "metadataList": "string[]",
                            "custodianList": "address[]",
                            "locationList": "string[]"
                        }
                    },
                    {
                        "path": "/assets/{asset_id}/transfer",
                        "method": "POST",
                        "description": "Transfer an asset to a new custodian",
                        "body": {
                            "to": "address",
                            "location": "string",
                            "proofHash": "bytes32"
                        }
                    },
                    {
                        "path": "/assets/{asset_id}/status",
                        "method": "PUT",
                        "description": "Update the status of an asset",
                        "body": {
                            "status": "integer (0-4)"
                        }
                    },
                    {
                        "path": "/analytics/custodian/{address}",
                        "method": "GET",
                        "description": "Get analytics for a specific custodian"
                    },
                    {
                        "path": "/analytics/global",
                        "method": "GET",
                        "description": "Get global supply chain analytics"
                    }
                ]
            }
        }
    })

# Real-time data streaming endpoint using Server-Sent Events
@app.route('/api/stream/assets', methods=['GET'])
def stream_assets():
    def event_stream():
        last_event_time = datetime.now()
        
        while True:
            # Check for new assets or updates every 5 seconds
            current_time = datetime.now()
            if (current_time - last_event_time).total_seconds() >= 5:
                try:
                    # This would be replaced with actual data from blockchain
                    data = {
                        "event": "asset_update",
                        "timestamp": current_time.isoformat(),
                        "data": {
                            "recentAssets": [],  # Would be populated with real data
                            "recentTransfers": []  # Would be populated with real data
                        }
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    last_event_time = current_time
                except Exception as e:
                    logger.error(f"Error in event stream: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return app.response_class(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
