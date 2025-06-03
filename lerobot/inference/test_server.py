#!/usr/bin/env python
"""
Test script that exactly matches your server's observation structure.
Use this to verify the WebSocket communication works with your real data.
"""

import asyncio
import json
import pickle
import base64
import gzip
import torch
import websockets
from collections import OrderedDict

class ExactStructureClient:
    def __init__(self, host="localhost", port=8765, compression_level=6):
        self.uri = f"ws://{host}:{port}"
        self.compression_level = compression_level
    
    def _serialize_tensor(self, tensor):
        """Serialize a tensor to compressed base64 string."""
        pickled_data = pickle.dumps(tensor)
        compressed_data = gzip.compress(pickled_data, compresslevel=self.compression_level)
        return base64.b64encode(compressed_data).decode('utf-8')
    
    def _deserialize_tensor(self, data):
        """Deserialize a tensor from compressed base64 string."""
        compressed_data = base64.b64decode(data.encode('utf-8'))
        pickled_data = gzip.decompress(compressed_data)
        return pickle.loads(pickled_data)
    
    def _serialize_observation(self, obs):
        """Serialize observation dictionary recursively."""
        result = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                result[key] = self._serialize_tensor(value)
            elif isinstance(value, dict):
                result[key] = {k: self._serialize_tensor(v) for k, v in value.items()}
            else:
                result[key] = value
        return result

def create_exact_observation():
    """Create observation with exact structure from your server."""
    # Using OrderedDict to match your server exactly
    observation = OrderedDict()
    
    # agent_pos: shape (1, 14)
    observation['agent_pos'] = torch.randn(1, 14)
    
    # pixels: nested dict with 'top' key, shape (1, 480, 640, 3)
    observation['pixels'] = {
        'top': torch.randint(0, 256, (1, 480, 640, 3), dtype=torch.uint8)
    }
    
    return observation

async def test_exact_structure():
    """Test with the exact observation structure from your server."""
    print("🎯 Testing with EXACT server observation structure...")
    print("=" * 60)
    
    client = ExactStructureClient()
    
    # Create observation matching your server exactly
    observation = create_exact_observation()
    
    print(f"📊 Observation structure:")
    print(f"   Type: {type(observation)}")
    print(f"   Keys: {list(observation.keys())}")
    print(f"   agent_pos shape: {observation['agent_pos'].shape}")
    print(f"   pixels keys: {list(observation['pixels'].keys())}")
    print(f"   pixels['top'] shape: {observation['pixels']['top'].shape}")
    print(f"   pixels['top'] dtype: {observation['pixels']['top'].dtype}")
    
    # Calculate data sizes
    agent_size = observation['agent_pos'].numel() * observation['agent_pos'].element_size()
    pixel_size = observation['pixels']['top'].numel() * observation['pixels']['top'].element_size()
    total_raw_size = agent_size + pixel_size
    
    print(f"\n📏 Data sizes:")
    print(f"   agent_pos: {agent_size} bytes")
    print(f"   pixels['top']: {pixel_size:,} bytes ({pixel_size / (1024*1024):.2f}MB)")
    print(f"   Total raw: {total_raw_size:,} bytes ({total_raw_size / (1024*1024):.2f}MB)")
    
    try:
        print(f"\n🔄 Connecting to server...")
        async with websockets.connect(
            client.uri, 
            max_size=100 * 1024 * 1024  # 100MB limit
        ) as ws:
            print("✅ Connected!")
            
            # Test 1: Ping
            print("\n🏓 Testing ping...")
            await ws.send(json.dumps({"type": "ping"}))
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "pong":
                print("✅ Ping successful!")
            else:
                print(f"❌ Ping failed: {data}")
                return
            
            # Test 2: Reset
            print("\n🔄 Testing reset...")
            await ws.send(json.dumps({"type": "reset"}))
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "reset_response":
                print("✅ Reset successful!")
            else:
                print(f"❌ Reset failed: {data}")
                return
            
            # Test 3: Action selection with exact structure
            print("\n🤖 Testing action selection with exact structure...")
            
            print("   🔄 Serializing observation...")
            serialized_obs = client._serialize_observation(observation)
            
            message = {
                "type": "select_action",
                "observation": serialized_obs
            }
            
            # Check serialized message size
            message_str = json.dumps(message)
            message_size = len(message_str.encode('utf-8'))
            compression_ratio = total_raw_size / message_size
            
            print(f"   📦 Serialized message size: {message_size:,} bytes ({message_size / (1024*1024):.2f}MB)")
            print(f"   🗜️  Compression ratio: {compression_ratio:.1f}x")
            
            print("   📤 Sending action request...")
            await ws.send(message_str)
            
            print("   📥 Waiting for server response...")
            response = await ws.recv()
            data = json.loads(response)
            
            if data.get("type") == "action_response":
                action = client._deserialize_tensor(data["action"])
                print(f"✅ SUCCESS! Received action from server!")
                print(f"   📈 Action shape: {action.shape}")
                print(f"   📊 Action dtype: {action.dtype}")
                print(f"   📊 Action range: [{action.min():.3f}, {action.max():.3f}]")
                print(f"   🎯 First 5 values: {action.flatten()[:5].tolist()}")
                
                return action
                
            elif data.get("type") == "error":
                print(f"❌ Server returned error: {data['message']}")
                return None
            else:
                print(f"❌ Unexpected response type: {data}")
                return None
                
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Is the server running on localhost:8765?")
        return None
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None

async def benchmark_compression():
    """Benchmark compression effectiveness."""
    print("\n🔬 Compression Benchmark")
    print("=" * 30)
    
    client = ExactStructureClient()
    observation = create_exact_observation()
    
    # Test different compression levels
    levels = [1, 3, 6, 9]
    
    for level in levels:
        client.compression_level = level
        
        # Serialize with this compression level
        start_time = asyncio.get_event_loop().time()
        serialized = client._serialize_observation(observation)
        serialize_time = asyncio.get_event_loop().time() - start_time
        
        # Calculate sizes
        message = {"type": "select_action", "observation": serialized}
        message_str = json.dumps(message)
        compressed_size = len(message_str.encode('utf-8'))
        
        # Calculate raw size
        raw_size = (observation['agent_pos'].numel() * observation['agent_pos'].element_size() +
                   observation['pixels']['top'].numel() * observation['pixels']['top'].element_size())
        
        ratio = raw_size / compressed_size
        
        print(f"   Level {level}: {compressed_size:,} bytes ({ratio:.1f}x ratio, {serialize_time*1000:.1f}ms)")
    
    print(f"\n💡 Recommendation: Use level 6 for good balance of speed vs compression")

if __name__ == "__main__":
    print("🧪 LeRobot WebSocket - Exact Structure Test")
    print("=" * 50)
    print("This test uses the EXACT observation structure from your server:")
    print("- OrderedDict with keys ['agent_pos', 'pixels']")
    print("- agent_pos: torch.Tensor shape (1, 14)")
    print("- pixels['top']: torch.Tensor shape (1, 480, 640, 3), dtype=uint8")
    print()
    
    async def run_all_tests():
        # Test the exact structure
        result = await test_exact_structure()
        
        if result is not None:
            # Run compression benchmark
            await benchmark_compression()
            print("\n🎉 All tests completed successfully!")
            print("\n✨ Your WebSocket server is working correctly with the exact observation structure!")
        else:
            print("\n❌ Tests failed. Check server logs for details.")
    
    asyncio.run(run_all_tests())