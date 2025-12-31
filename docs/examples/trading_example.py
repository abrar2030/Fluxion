# Trading Example - Complete Workflow

This example demonstrates a complete trading workflow using Fluxion, from connecting to the API to executing trades and monitoring positions.

## Prerequisites

- Fluxion backend running at `http://localhost:8000`
- Python 3.10+ installed
- Valid API credentials

## Installation

```bash
pip install httpx eth-account python-dotenv
```

## Complete Trading Example

```python
#!/usr/bin/env python3
"""
Complete trading workflow example for Fluxion
Demonstrates authentication, market data, order placement, and monitoring.
"""

import asyncio
import os
from typing import Dict, Optional
from decimal import Decimal

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")


class FluxionTrader:
    """Fluxion trading client"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def login(self, username: str, password: str) -> Dict:
        """Authenticate and get JWT token"""
        response = await self.client.post(
            f"{self.base_url}/api/v1/auth/login",
            json={
                "username": username,
                "password": password
            }
        )
        response.raise_for_status()
        data = response.json()
        self.token = data["access_token"]
        print(f"✓ Authenticated as {data['user']['username']}")
        return data
    
    def _headers(self) -> Dict:
        """Get authorization headers"""
        return {"Authorization": f"Bearer {self.token}"}
    
    async def get_market_price(self, asset_id: str) -> Decimal:
        """Get current market price for asset"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/market/price/{asset_id}",
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        price = Decimal(data["price"])
        print(f"✓ {asset_id} current price: ${price:,.2f}")
        return price
    
    async def get_account_balance(self) -> Dict:
        """Get account balance"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/account/balance",
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ Account balance: ${Decimal(data['usd_balance']):,.2f}")
        return data
    
    async def place_market_order(
        self,
        asset_id: str,
        side: str,
        amount: str,
        slippage_tolerance: str = "0.01"
    ) -> Dict:
        """
        Place a market order
        
        Args:
            asset_id: Asset to trade (e.g., "BTC-USD")
            side: "BUY" or "SELL"
            amount: Amount to trade (decimal string)
            slippage_tolerance: Max slippage (default 1%)
            
        Returns:
            Order details
        """
        order_data = {
            "asset_id": asset_id,
            "side": side,
            "amount": amount,
            "order_type": "MARKET",
            "slippage_tolerance": slippage_tolerance
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/order",
            json=order_data,
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Order placed: {data['order_id']}")
        print(f"  {side} {amount} {asset_id}")
        print(f"  Status: {data['status']}")
        print(f"  Estimated price: ${Decimal(data.get('estimated_price', 0)):,.2f}")
        
        return data
    
    async def place_limit_order(
        self,
        asset_id: str,
        side: str,
        amount: str,
        price: str
    ) -> Dict:
        """Place a limit order"""
        order_data = {
            "asset_id": asset_id,
            "side": side,
            "amount": amount,
            "price": price,
            "order_type": "LIMIT"
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/v1/order",
            json=order_data,
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Limit order placed: {data['order_id']}")
        print(f"  {side} {amount} {asset_id} @ ${Decimal(price):,.2f}")
        
        return data
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Check order status"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/order/{order_id}",
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Order {order_id} status: {data['status']}")
        if data['status'] == 'FILLED':
            print(f"  Filled at: ${Decimal(data['average_price']):,.2f}")
            print(f"  Fees: ${Decimal(data['fees']):,.2f}")
        
        return data
    
    async def get_open_orders(self) -> list:
        """Get all open orders"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/orders?status=PENDING",
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Open orders: {len(data['orders'])}")
        for order in data['orders']:
            print(f"  {order['order_id']}: {order['side']} {order['amount']} "
                  f"{order['asset_id']} @ ${Decimal(order.get('price', 0)):,.2f}")
        
        return data['orders']
    
    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel an open order"""
        response = await self.client.delete(
            f"{self.base_url}/api/v1/order/{order_id}",
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Order {order_id} cancelled")
        return data
    
    async def get_portfolio(self) -> Dict:
        """Get portfolio summary"""
        response = await self.client.get(
            f"{self.base_url}/api/v1/portfolio",
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Portfolio value: ${Decimal(data['total_value']):,.2f}")
        print(f"  P&L: ${Decimal(data['unrealized_pnl']):,.2f} "
              f"({Decimal(data['pnl_percentage']):.2f}%)")
        
        for position in data['positions']:
            print(f"  {position['asset_id']}: {position['amount']} "
                  f"(${Decimal(position['value']):,.2f})")
        
        return data
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def main():
    """Main trading workflow example"""
    
    print("=" * 60)
    print("Fluxion Trading Example")
    print("=" * 60)
    print()
    
    # Initialize trader
    trader = FluxionTrader(BASE_URL)
    
    try:
        # 1. Login
        print("Step 1: Authentication")
        print("-" * 60)
        await trader.login(USERNAME, PASSWORD)
        print()
        
        # 2. Check balance
        print("Step 2: Check Account Balance")
        print("-" * 60)
        balance = await trader.get_account_balance()
        print()
        
        # 3. Get market price
        print("Step 3: Get Market Price")
        print("-" * 60)
        asset_id = "BTC-USD"
        price = await trader.get_market_price(asset_id)
        print()
        
        # 4. Place market order (buy)
        print("Step 4: Place Market Order (Buy)")
        print("-" * 60)
        buy_order = await trader.place_market_order(
            asset_id=asset_id,
            side="BUY",
            amount="0.1",
            slippage_tolerance="0.01"
        )
        print()
        
        # 5. Wait and check order status
        print("Step 5: Check Order Status")
        print("-" * 60)
        await asyncio.sleep(2)  # Wait for execution
        order_status = await trader.get_order_status(buy_order['order_id'])
        print()
        
        # 6. Place limit order (sell)
        print("Step 6: Place Limit Order (Sell)")
        print("-" * 60)
        sell_price = str(float(price) * 1.05)  # 5% above current price
        sell_order = await trader.place_limit_order(
            asset_id=asset_id,
            side="SELL",
            amount="0.1",
            price=sell_price
        )
        print()
        
        # 7. Check open orders
        print("Step 7: View Open Orders")
        print("-" * 60)
        open_orders = await trader.get_open_orders()
        print()
        
        # 8. Get portfolio
        print("Step 8: View Portfolio")
        print("-" * 60)
        portfolio = await trader.get_portfolio()
        print()
        
        # 9. Cancel limit order (optional)
        print("Step 9: Cancel Limit Order")
        print("-" * 60)
        if open_orders:
            await trader.cancel_order(sell_order['order_id'])
        print()
        
        print("=" * 60)
        print("Trading workflow completed successfully!")
        print("=" * 60)
        
    except httpx.HTTPStatusError as e:
        print(f"✗ HTTP Error: {e.response.status_code}")
        print(f"  {e.response.json()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await trader.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Expected Output

```
============================================================
Fluxion Trading Example
============================================================

Step 1: Authentication
------------------------------------------------------------
✓ Authenticated as trader@example.com

Step 2: Check Account Balance
------------------------------------------------------------
✓ Account balance: $100,000.00

Step 3: Get Market Price
------------------------------------------------------------
✓ BTC-USD current price: $50,125.50

Step 4: Place Market Order (Buy)
------------------------------------------------------------
✓ Order placed: ord_1234567890
  BUY 0.1 BTC-USD
  Status: PENDING
  Estimated price: $50,150.25

Step 5: Check Order Status
------------------------------------------------------------
✓ Order ord_1234567890 status: FILLED
  Filled at: $50,150.25
  Fees: $25.08

Step 6: Place Limit Order (Sell)
------------------------------------------------------------
✓ Limit order placed: ord_0987654321
  SELL 0.1 BTC-USD @ $52,631.78

Step 7: View Open Orders
------------------------------------------------------------
✓ Open orders: 1
  ord_0987654321: SELL 0.1 BTC-USD @ $52,631.78

Step 8: View Portfolio
------------------------------------------------------------
✓ Portfolio value: $100,000.00
  P&L: $0.00 (0.00%)
  BTC-USD: 0.1 ($5,015.03)

Step 9: Cancel Limit Order
------------------------------------------------------------
✓ Order ord_0987654321 cancelled

============================================================
Trading workflow completed successfully!
============================================================
```

## Environment Variables

Create a `.env` file:

```bash
# API Configuration
API_URL=http://localhost:8000
USERNAME=your-email@example.com
PASSWORD=your-password

# Optional: API Key for additional authentication
API_KEY=your-api-key
```

## Advanced Examples

### Using TWAP Orders

```python
# Time-Weighted Average Price order
twap_order = await trader.place_twap_order(
    asset_id="BTC-USD",
    side="BUY",
    amount="1.0",
    duration_minutes=60,  # Execute over 1 hour
    num_slices=12  # 12 sub-orders (one every 5 minutes)
)
```

### Risk Management

```python
# Get risk metrics before trading
risk = await trader.get_risk_metrics()
if risk['liquidation_risk'] != 'LOW':
    print("Warning: High liquidation risk!")
    return

# Set stop-loss
stop_loss_order = await trader.place_limit_order(
    asset_id="BTC-USD",
    side="SELL",
    amount="0.1",
    price=str(float(entry_price) * 0.95)  # 5% below entry
)
```

### Monitoring with WebSocket

```python
import websockets
import json

async def monitor_prices():
    uri = "ws://localhost:8000/ws/v1"
    async with websockets.connect(uri) as websocket:
        # Subscribe to price updates
        await websocket.send(json.dumps({
            "type": "subscribe",
            "channels": ["trades"],
            "symbols": ["BTC-USD", "ETH-USD"]
        }))
        
        # Listen for updates
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'trade':
                print(f"{data['symbol']}: ${data['price']}")
```

## Error Handling

```python
try:
    order = await trader.place_market_order(...)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 400:
        error = e.response.json()
        if error['error_code'] == 'INSUFFICIENT_BALANCE':
            print("Insufficient balance for trade")
        elif error['error_code'] == 'INVALID_AMOUNT':
            print("Invalid trade amount")
    elif e.response.status_code == 429:
        print("Rate limit exceeded, wait before retrying")
```

## Next Steps

- Read [API Reference](../API.md) for complete endpoint documentation
- See [Usage Guide](../USAGE.md) for more examples
- Check [Risk Management](../API.md#risk-management) for portfolio protection
- Explore [AI Predictions](ml_prediction.py) for data-driven trading

## Disclaimer

This example is for educational purposes. Always test with small amounts first and understand the risks before trading real assets.
