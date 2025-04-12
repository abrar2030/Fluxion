import { ethers } from 'ethers';

export const SYNTHETIC_ABI = [/* Full ABI */];

export async function createSyntheticAsset(provider, params) {
  const signer = provider.getSigner();
  const factory = new ethers.Contract(
    process.env.FACTORY_ADDRESS,
    SYNTHETIC_ABI,
    signer
  );
  
  const tx = await factory.createSynthetic(
    params.assetId,
    params.oracle,
    params.jobId,
    ethers.utils.parseEther(params.payment)
  );
  
  return tx.wait();
}