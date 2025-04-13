# API Specification

## Base URL
```
https://api.fluxion.exchange/v1
```

## Authentication
All API endpoints require authentication using JWT tokens. Include the token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### Trading Operations

#### Create Synthetic Asset Order
```http
POST /order
```

**Request Body:**
```json
{
  "asset_id": "string",
  "side": "BUY" | "SELL",
  "amount": "string",
  "price": "string",
  "order_type": "MARKET" | "LIMIT" | "TWAP" | "VWAP",
  "slippage_tolerance": "string"
}
```

**Response:**
```json
{
  "order_id": "string",
  "status": "PENDING",
  "timestamp": "string"
}
```

#### Get Order Status
```http
GET /order/{order_id}
```

### Liquidity Operations

#### Get Pool Statistics
```http
GET /pool/{pool_id}/stats
```

**Response:**
```json
{
  "tvl": "string",
  "volume_24h": "string",
  "apy": "string",
  "utilization": "string"
}
```

#### Add Liquidity
```http
POST /pool/{pool_id}/deposit
```

### Risk Management

#### Get Risk Metrics
```http
GET /risk/metrics
```

**Response:**
```json
{
  "var_95": "string",
  "expected_shortfall": "string",
  "correlation_matrix": "object",
  "volatility": "string"
}
```

### AI Model Endpoints

#### Get Price Predictions
```http
GET /ai/predictions/{asset_id}
```

**Response:**
```json
{
  "predictions": [
    {
      "timestamp": "string",
      "price": "string",
      "confidence": "number"
    }
  ]
}
```

## Rate Limits
- 100 requests per minute for regular endpoints
- 1000 requests per minute for market data endpoints
- 10 requests per minute for AI prediction endpoints

## Error Codes
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Websocket API

### Connection
```
wss://api.fluxion.exchange/ws/v1
```

### Market Data Stream
```json
{
  "type": "subscribe",
  "channels": ["trades", "orderbook", "liquidations"],
  "symbols": ["BTC-USD", "ETH-USD"]
}
```

## SDK Examples
```python
from fluxion_sdk import FluxionClient

client = FluxionClient(api_key="your_api_key")

# Place a market order
order = client.create_order(
    asset_id="BTC-USD",
    side="BUY",
    amount="1.0",
    order_type="MARKET"
)
```
