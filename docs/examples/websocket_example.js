# WebSocket Real-Time Data Example

This example demonstrates how to connect to Fluxion's WebSocket API for real-time market data, order updates, and liquidation events.

## Prerequisites

- Node.js 18+ or modern browser
- Fluxion backend running with WebSocket support

## Installation (Node.js)

```bash
npm install ws
```

## Basic WebSocket Connection

```javascript
// websocket_example.js
const WebSocket = require('ws');

// Connect to Fluxion WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/v1');

ws.on('open', () => {
    console.log('✓ Connected to Fluxion WebSocket');
    
    // Subscribe to channels
    ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['trades', 'orderbook', 'liquidations'],
        symbols: ['BTC-USD', 'ETH-USD']
    }));
});

ws.on('message', (data) => {
    const message = JSON.parse(data);
    
    switch (message.type) {
        case 'trade':
            console.log(`Trade: ${message.symbol} ${message.side} ${message.amount} @ $${message.price}`);
            break;
            
        case 'orderbook':
            console.log(`OrderBook Update: ${message.symbol}`);
            console.log(`  Best Bid: $${message.bids[0][0]}`);
            console.log(`  Best Ask: $${message.asks[0][0]}`);
            break;
            
        case 'liquidation':
            console.log(`⚠️  Liquidation: ${message.position_id} ${message.symbol} ${message.amount}`);
            break;
            
        default:
            console.log('Unknown message type:', message.type);
    }
});

ws.on('error', (error) => {
    console.error('WebSocket error:', error);
});

ws.on('close', () => {
    console.log('WebSocket disconnected');
});

// Graceful shutdown
process.on('SIGINT', () => {
    ws.close();
    process.exit();
});
```

## React Hook for WebSocket

```javascript
// useFluxionWebSocket.js
import { useEffect, useRef, useState } from 'react';

/**
 * Custom hook for Fluxion WebSocket connection
 * @param {string} url - WebSocket URL
 * @param {Object} subscription - Subscription configuration
 * @returns {Object} WebSocket state and data
 */
export const useFluxionWebSocket = (url, subscription) => {
    const [connected, setConnected] = useState(false);
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);

    useEffect(() => {
        // Create WebSocket connection
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('WebSocket connected');
            setConnected(true);
            setError(null);

            // Send subscription message
            ws.send(JSON.stringify({
                type: 'subscribe',
                ...subscription
            }));
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                setData(message);
            } catch (err) {
                console.error('Failed to parse message:', err);
            }
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            setError(err);
            setConnected(false);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            setConnected(false);
        };

        // Cleanup on unmount
        return () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, [url, JSON.stringify(subscription)]);

    // Function to send messages
    const send = (message) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        }
    };

    return { connected, data, error, send };
};
```

## React Component Example

```javascript
// PriceTracker.jsx
import React, { useState, useEffect } from 'react';
import { useFluxionWebSocket } from './useFluxionWebSocket';

export const PriceTracker = ({ symbols = ['BTC-USD', 'ETH-USD'] }) => {
    const [prices, setPrices] = useState({});
    
    const { connected, data, error } = useFluxionWebSocket(
        'ws://localhost:8000/ws/v1',
        {
            channels: ['trades'],
            symbols: symbols
        }
    );

    useEffect(() => {
        if (data && data.type === 'trade') {
            setPrices(prev => ({
                ...prev,
                [data.symbol]: {
                    price: data.price,
                    change: calculateChange(prev[data.symbol]?.price, data.price),
                    timestamp: data.timestamp
                }
            }));
        }
    }, [data]);

    const calculateChange = (oldPrice, newPrice) => {
        if (!oldPrice) return 0;
        return ((newPrice - oldPrice) / oldPrice * 100).toFixed(2);
    };

    if (error) {
        return <div className="error">WebSocket Error: {error.message}</div>;
    }

    if (!connected) {
        return <div className="loading">Connecting...</div>;
    }

    return (
        <div className="price-tracker">
            <h2>Live Prices</h2>
            {Object.entries(prices).map(([symbol, data]) => (
                <div key={symbol} className="price-card">
                    <div className="symbol">{symbol}</div>
                    <div className="price">${data.price.toLocaleString()}</div>
                    <div className={`change ${data.change >= 0 ? 'positive' : 'negative'}`}>
                        {data.change >= 0 ? '+' : ''}{data.change}%
                    </div>
                    <div className="timestamp">
                        {new Date(data.timestamp).toLocaleTimeString()}
                    </div>
                </div>
            ))}
        </div>
    );
};
```

## Order Book Visualization

```javascript
// OrderBook.jsx
import React, { useState, useEffect } from 'react';
import { useFluxionWebSocket } from './useFluxionWebSocket';

export const OrderBook = ({ symbol = 'BTC-USD' }) => {
    const [orderBook, setOrderBook] = useState({ bids: [], asks: [] });
    
    const { connected, data } = useFluxionWebSocket(
        'ws://localhost:8000/ws/v1',
        {
            channels: ['orderbook'],
            symbols: [symbol]
        }
    );

    useEffect(() => {
        if (data && data.type === 'orderbook' && data.symbol === symbol) {
            setOrderBook({
                bids: data.bids.slice(0, 10),  // Top 10 bids
                asks: data.asks.slice(0, 10)   // Top 10 asks
            });
        }
    }, [data, symbol]);

    const renderRow = (price, amount, side) => {
        const [priceStr, amountStr] = [price, amount];
        const total = (parseFloat(priceStr) * parseFloat(amountStr)).toFixed(2);
        
        return (
            <tr className={side}>
                <td className="price">{parseFloat(priceStr).toLocaleString()}</td>
                <td className="amount">{parseFloat(amountStr).toFixed(4)}</td>
                <td className="total">${parseFloat(total).toLocaleString()}</td>
            </tr>
        );
    };

    if (!connected) {
        return <div>Connecting to order book...</div>;
    }

    return (
        <div className="orderbook">
            <h3>Order Book - {symbol}</h3>
            
            <div className="asks">
                <h4>Asks (Sell Orders)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Price</th>
                            <th>Amount</th>
                            <th>Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {orderBook.asks.reverse().map(([price, amount], i) => 
                            renderRow(price, amount, 'ask')
                        )}
                    </tbody>
                </table>
            </div>
            
            <div className="spread">
                <div className="spread-value">
                    Spread: ${(
                        parseFloat(orderBook.asks[0]?.[0] || 0) - 
                        parseFloat(orderBook.bids[0]?.[0] || 0)
                    ).toFixed(2)}
                </div>
            </div>
            
            <div className="bids">
                <h4>Bids (Buy Orders)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Price</th>
                            <th>Amount</th>
                            <th>Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {orderBook.bids.map(([price, amount], i) => 
                            renderRow(price, amount, 'bid')
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
```

## Advanced: Trading Bot with WebSocket

```javascript
// simpleBot.js
const WebSocket = require('ws');

class FluxionTradingBot {
    constructor(wsUrl, apiUrl, apiKey) {
        this.wsUrl = wsUrl;
        this.apiUrl = apiUrl;
        this.apiKey = apiKey;
        this.ws = null;
        this.prices = {};
        this.positions = {};
    }

    connect() {
        this.ws = new WebSocket(this.wsUrl);

        this.ws.on('open', () => {
            console.log('Bot connected to Fluxion');
            
            // Subscribe to trades and liquidations
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                channels: ['trades', 'liquidations'],
                symbols: ['BTC-USD', 'ETH-USD']
            }));
        });

        this.ws.on('message', async (data) => {
            const message = JSON.parse(data);
            
            if (message.type === 'trade') {
                await this.handleTrade(message);
            } else if (message.type === 'liquidation') {
                await this.handleLiquidation(message);
            }
        });

        this.ws.on('error', (error) => {
            console.error('WebSocket error:', error);
        });

        this.ws.on('close', () => {
            console.log('Connection closed, reconnecting in 5s...');
            setTimeout(() => this.connect(), 5000);
        });
    }

    async handleTrade(message) {
        const { symbol, price, amount, side } = message;
        
        // Update price tracking
        this.prices[symbol] = price;
        
        // Simple moving average strategy (example)
        const avgPrice = this.calculateSMA(symbol, 20);
        
        if (price < avgPrice * 0.98) {
            // Price 2% below average - buy signal
            await this.placeBuyOrder(symbol, '0.01');
        } else if (price > avgPrice * 1.02) {
            // Price 2% above average - sell signal
            await this.placeSellOrder(symbol, '0.01');
        }
    }

    async handleLiquidation(message) {
        console.log(`⚠️  Liquidation detected: ${message.symbol}`);
        // Could implement liquidation hunting strategy
    }

    calculateSMA(symbol, period) {
        // Simplified - in production, maintain price history
        return this.prices[symbol] || 0;
    }

    async placeBuyOrder(symbol, amount) {
        try {
            const response = await fetch(`${this.apiUrl}/api/v1/order`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    asset_id: symbol,
                    side: 'BUY',
                    amount: amount,
                    order_type: 'MARKET'
                })
            });
            
            const order = await response.json();
            console.log(`✓ Buy order placed: ${order.order_id}`);
        } catch (error) {
            console.error('Failed to place buy order:', error);
        }
    }

    async placeSellOrder(symbol, amount) {
        // Similar to placeBuyOrder but with side='SELL'
        // Implementation omitted for brevity
    }

    start() {
        console.log('Starting Fluxion trading bot...');
        this.connect();
    }
}

// Usage
const bot = new FluxionTradingBot(
    'ws://localhost:8000/ws/v1',
    'http://localhost:8000',
    process.env.API_KEY
);

bot.start();
```

## Subscription Options

### Available Channels

| Channel | Description | Message Format |
|---------|-------------|----------------|
| `trades` | Real-time trade executions | `{type: 'trade', symbol, price, amount, side, timestamp}` |
| `orderbook` | Order book updates | `{type: 'orderbook', symbol, bids: [[price, amount]], asks: [[price, amount]]}` |
| `liquidations` | Liquidation events | `{type: 'liquidation', position_id, symbol, amount, price}` |
| `user_orders` | User's order updates (requires auth) | `{type: 'order_update', order_id, status, ...}` |

### Subscription Message

```javascript
{
    "type": "subscribe",
    "channels": ["trades", "orderbook"],
    "symbols": ["BTC-USD", "ETH-USD"]
}
```

### Unsubscribe

```javascript
{
    "type": "unsubscribe",
    "channels": ["trades"],
    "symbols": ["BTC-USD"]
}
```

## Error Handling

```javascript
ws.on('error', (error) => {
    if (error.code === 'ECONNREFUSED') {
        console.error('Cannot connect to WebSocket server');
    } else if (error.code === 'ETIMEDOUT') {
        console.error('Connection timed out');
    } else {
        console.error('WebSocket error:', error);
    }
});

// Implement reconnection logic
function connectWithRetry(url, maxRetries = 5) {
    let retries = 0;
    
    function connect() {
        const ws = new WebSocket(url);
        
        ws.on('error', () => {
            if (retries < maxRetries) {
                retries++;
                console.log(`Retrying connection (${retries}/${maxRetries})...`);
                setTimeout(connect, 1000 * Math.pow(2, retries)); // Exponential backoff
            } else {
                console.error('Max retries reached');
            }
        });
        
        ws.on('open', () => {
            retries = 0;  // Reset on successful connection
            console.log('Connected successfully');
        });
        
        return ws;
    }
    
    return connect();
}
```

## Performance Tips

1. **Throttle updates**: For high-frequency data, throttle UI updates
```javascript
import { throttle } from 'lodash';

const throttledUpdate = throttle((data) => {
    setOrderBook(data);
}, 100); // Update max once per 100ms
```

2. **Batch updates**: Accumulate updates and process in batches
```javascript
let updateQueue = [];
setInterval(() => {
    if (updateQueue.length > 0) {
        processUpdates(updateQueue);
        updateQueue = [];
    }
}, 100);
```

3. **Connection pooling**: Reuse WebSocket connections
```javascript
const wsPool = {
    connections: {},
    get(url) {
        if (!this.connections[url]) {
            this.connections[url] = new WebSocket(url);
        }
        return this.connections[url];
    }
};
```

## Next Steps

- See [API Reference](../API.md#websocket-api) for complete WebSocket documentation
- Check [Trading Example](trading_example.py) for REST API integration
- Read [Usage Guide](../USAGE.md) for more patterns

## Browser Console Testing

Open browser console on any page and test WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/v1');

ws.onopen = () => {
    console.log('Connected');
    ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['trades'],
        symbols: ['BTC-USD']
    }));
};

ws.onmessage = (event) => {
    console.log('Received:', JSON.parse(event.data));
};
```
