import React, { useState } from 'react';
import { useWeb3 } from .../lib/web3-config.jsx';
import AssetSelector from './AssetSelector';
import WeightEditor from './WeightEditor';

export default function PoolCreator() {  
  const { contracts } = useWeb3();  
  const [poolConfig, setPoolConfig] = useState({  
    assets: [],  
    weights: [],  
    fee: 0.003  
  });  

  const createPool = async () => {  
    const tx = await contracts.factory.createPool(  
      poolConfig.assets,  
      poolConfig.weights.map(w => w * 1e18),  
      Math.floor(poolConfig.fee * 1e6)  
    );  
    await tx.wait();  
  };  

  return (  
    <div className="pool-creator-container">  
      <h2>Create New Liquidity Pool</h2>
      <AssetSelector onChange={assets => setPoolConfig({...poolConfig, assets})} />  
      <WeightEditor weights={poolConfig.weights} onChange={weights => setPoolConfig({...poolConfig, weights})} />  
      <button className="create-pool-btn" onClick={createPool}>Create Pool</button>  
    </div>  
  );  
}
