from web3 import Web3
import json
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainConnector:
    """
    Connector class for interacting with blockchain smart contracts
    """
    
    def __init__(self):
        # Initialize Web3 connection
        self.rpc_url = os.getenv('BLOCKCHAIN_RPC_URL', 'http://localhost:8545')
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Check connection
        if not self.w3.is_connected():
            logger.warning(f"Failed to connect to blockchain at {self.rpc_url}")
        
        # Load contract addresses
        self.supply_chain_address = os.getenv('SUPPLY_CHAIN_ADDRESS')
        self.liquidity_pool_address = os.getenv('LIQUIDITY_POOL_ADDRESS')
        self.synthetic_asset_address = os.getenv('SYNTHETIC_ASSET_ADDRESS')
        
        # Load contract ABIs
        self.load_contract_abis()
        
        # Initialize contract instances
        self.initialize_contracts()
        
        # Set default account
        self.default_account = os.getenv('DEFAULT_ACCOUNT')
        if self.default_account:
            self.w3.eth.default_account = self.default_account
    
    def load_contract_abis(self):
        """Load contract ABIs from JSON files"""
        try:
            # Supply Chain Tracker ABI
            supply_chain_path = os.path.join(os.path.dirname(__file__), '../contracts/SupplyChainTracker.json')
            if os.path.exists(supply_chain_path):
                with open(supply_chain_path, 'r') as f:
                    contract_data = json.load(f)
                    self.supply_chain_abi = contract_data['abi']
            else:
                logger.warning(f"Supply chain contract ABI file not found at {supply_chain_path}")
                self.supply_chain_abi = []
            
            # Liquidity Pool Manager ABI
            liquidity_pool_path = os.path.join(os.path.dirname(__file__), '../contracts/EnhancedLiquidityPoolManager.json')
            if os.path.exists(liquidity_pool_path):
                with open(liquidity_pool_path, 'r') as f:
                    contract_data = json.load(f)
                    self.liquidity_pool_abi = contract_data['abi']
            else:
                logger.warning(f"Liquidity pool contract ABI file not found at {liquidity_pool_path}")
                self.liquidity_pool_abi = []
            
            # Synthetic Asset Factory ABI
            synthetic_asset_path = os.path.join(os.path.dirname(__file__), '../contracts/SyntheticAssetFactory.json')
            if os.path.exists(synthetic_asset_path):
                with open(synthetic_asset_path, 'r') as f:
                    contract_data = json.load(f)
                    self.synthetic_asset_abi = contract_data['abi']
            else:
                logger.warning(f"Synthetic asset contract ABI file not found at {synthetic_asset_path}")
                self.synthetic_asset_abi = []
                
        except Exception as e:
            logger.error(f"Error loading contract ABIs: {e}")
    
    def initialize_contracts(self):
        """Initialize contract instances"""
        try:
            # Supply Chain Tracker contract
            if self.supply_chain_address and self.supply_chain_abi:
                self.supply_chain_contract = self.w3.eth.contract(
                    address=self.supply_chain_address,
                    abi=self.supply_chain_abi
                )
            else:
                self.supply_chain_contract = None
                logger.warning("Supply chain contract not initialized")
            
            # Liquidity Pool Manager contract
            if self.liquidity_pool_address and self.liquidity_pool_abi:
                self.liquidity_pool_contract = self.w3.eth.contract(
                    address=self.liquidity_pool_address,
                    abi=self.liquidity_pool_abi
                )
            else:
                self.liquidity_pool_contract = None
                logger.warning("Liquidity pool contract not initialized")
            
            # Synthetic Asset Factory contract
            if self.synthetic_asset_address and self.synthetic_asset_abi:
                self.synthetic_asset_contract = self.w3.eth.contract(
                    address=self.synthetic_asset_address,
                    abi=self.synthetic_asset_abi
                )
            else:
                self.synthetic_asset_contract = None
                logger.warning("Synthetic asset contract not initialized")
                
        except Exception as e:
            logger.error(f"Error initializing contracts: {e}")
    
    def get_asset_count(self):
        """Get the total number of assets in the supply chain"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        return self.supply_chain_contract.functions.getAssetCount().call()
    
    def get_asset(self, asset_id):
        """Get details for a specific asset"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        asset_data = self.supply_chain_contract.functions.getAsset(asset_id).call()
        
        # Format asset data
        asset = {
            'id': asset_data[0],
            'metadata': asset_data[1],
            'currentCustodian': asset_data[2],
            'timestamp': asset_data[3],
            'status': asset_data[4],
            'location': asset_data[5],
            'transferCount': asset_data[6]
        }
        
        return asset
    
    def get_transfer(self, asset_id, transfer_id):
        """Get transfer details for an asset"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        transfer_data = self.supply_chain_contract.functions.getTransfer(asset_id, transfer_id).call()
        
        # Format transfer data
        transfer = {
            'from': transfer_data[0],
            'to': transfer_data[1],
            'timestamp': transfer_data[2],
            'location': transfer_data[3],
            'proofHash': transfer_data[4].hex()
        }
        
        return transfer
    
    def get_custodian_assets(self, custodian):
        """Get all assets for a custodian"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        asset_ids = self.supply_chain_contract.functions.getCustodianAssets(custodian).call()
        
        assets = []
        for asset_id in asset_ids:
            try:
                asset = self.get_asset(asset_id)
                assets.append(asset)
            except Exception as e:
                logger.warning(f"Error fetching asset {asset_id}: {e}")
        
        return assets
    
    def create_asset(self, metadata, initial_custodian, location):
        """Create a new asset in the supply chain"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        # Build transaction
        tx = self.supply_chain_contract.functions.createAsset(
            metadata,
            initial_custodian,
            location
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.transactionHash.hex()
    
    def batch_create_assets(self, metadata_list, custodian_list, location_list):
        """Create multiple assets in a single transaction"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        # Build transaction
        tx = self.supply_chain_contract.functions.batchCreateAssets(
            metadata_list,
            custodian_list,
            location_list
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            'gas': 5000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.transactionHash.hex()
    
    def transfer_asset(self, asset_id, to_address, location, proof_hash):
        """Transfer an asset to a new custodian"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        # Convert proof hash to bytes32 if it's a string
        if isinstance(proof_hash, str):
            if proof_hash.startswith('0x'):
                proof_hash = bytes.fromhex(proof_hash[2:])
            else:
                proof_hash = bytes.fromhex(proof_hash)
        
        # Build transaction
        tx = self.supply_chain_contract.functions.transferAsset(
            asset_id,
            to_address,
            location,
            proof_hash
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.transactionHash.hex()
    
    def update_asset_status(self, asset_id, status):
        """Update the status of an asset"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        # Build transaction
        tx = self.supply_chain_contract.functions.updateAssetStatus(
            asset_id,
            status
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            'gas': 1000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.transactionHash.hex()
    
    def request_location_update(self, asset_id, oracle, job_id, fee):
        """Request a location update for an asset via Chainlink oracle"""
        if not self.supply_chain_contract:
            raise Exception("Supply chain contract not initialized")
        
        # Convert job_id to bytes32 if it's a string
        if isinstance(job_id, str):
            if job_id.startswith('0x'):
                job_id = bytes.fromhex(job_id[2:])
            else:
                job_id = bytes.fromhex(job_id)
        
        # Build transaction
        tx = self.supply_chain_contract.functions.requestLocationUpdate(
            asset_id,
            oracle,
            job_id,
            fee
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.transactionHash.hex()
    
    # Liquidity Pool Manager methods
    
    def create_pool(self, assets, weights, fee, amplification, oracles, heartbeats):
        """Create a new liquidity pool"""
        if not self.liquidity_pool_contract:
            raise Exception("Liquidity pool contract not initialized")
        
        # Build transaction
        tx = self.liquidity_pool_contract.functions.createPool(
            assets,
            weights,
            fee,
            amplification,
            oracles,
            heartbeats
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.transactionHash.hex()
    
    def get_pool(self, pool_id):
        """Get pool configuration"""
        if not self.liquidity_pool_contract:
            raise Exception("Liquidity pool contract not initialized")
        
        pool_data = self.liquidity_pool_contract.functions.getPool(pool_id).call()
        
        # Format pool data
        pool = {
            'assets': pool_data[0],
            'weights': [int(w) for w in pool_data[1]],
            'fee': int(pool_data[2]),
            'amplification': int(pool_data[3]),
            'active': pool_data[4],
            'totalLiquidity': int(pool_data[5])
        }
        
        return pool
    
    # Synthetic Asset Factory methods
    
    def create_synthetic(self, asset_id, oracle, job_id, fee):
        """Create a new synthetic asset"""
        if not self.synthetic_asset_contract:
            raise Exception("Synthetic asset contract not initialized")
        
        # Convert job_id to bytes32 if it's a string
        if isinstance(job_id, str):
            if job_id.startswith('0x'):
                job_id = bytes.fromhex(job_id[2:])
            else:
                job_id = bytes.fromhex(job_id)
        
        # Convert asset_id to bytes32 if it's a string
        if isinstance(asset_id, str):
            if asset_id.startswith('0x'):
                asset_id = bytes.fromhex(asset_id[2:])
            else:
                asset_id = bytes.fromhex(asset_id)
        
        # Build transaction
        tx = self.synthetic_asset_contract.functions.createSynthetic(
            asset_id,
            oracle,
            job_id,
            fee
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'nonce': self.w3.eth.get_transaction_count(self.w3.eth.default_account),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=os.getenv('PRIVATE_KEY'))
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.transactionHash.hex()
