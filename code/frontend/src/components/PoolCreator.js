import { useWeb3 } from './web3-config';  

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
    <div>  
      <AssetSelector onChange={assets => setPoolConfig({...poolConfig, assets})} />  
      <WeightEditor weights={poolConfig.weights} onChange={weights => setPoolConfig({...poolConfig, weights})} />  
      <button onClick={createPool}>Create Pool</button>  
    </div>  
  );  
}  