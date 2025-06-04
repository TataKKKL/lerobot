#!/usr/bin/env python
"""
Usage examples for the LeRobot WebSocket Client API.

This file demonstrates different ways to use the LeRobotClient class
in your projects.
"""

import asyncio
import numpy as np
import logging
from collections import OrderedDict
from lerobot_client import LeRobotClient, LeRobotClientError, create_client_and_connect

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

def create_sample_observation():
    """Create a sample observation matching the expected structure."""
    observation = OrderedDict()
    
    # agent_pos: shape (1, 14)
    observation['agent_pos'] = np.random.randn(1, 14).astype(np.float32)
    
    # pixels: nested dict with 'top' key, shape (1, 480, 640, 3)
    observation['pixels'] = {
        'top': np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
    }
    
    return observation

async def example_1_basic_usage():
    """Example 1: Basic usage with manual connection management."""
    print("Example 1: Basic Usage")
    print("-" * 30)
    
    # Create client
    client = LeRobotClient(host="localhost", port=8765)
    
    try:
        # Connect to server
        await client.connect()
        
        # Test connection
        if await client.ping():
            print("✅ Server is responding")
        else:
            print("❌ Server not responding")
            return
        
        # Reset environment
        await client.reset()
        print("✅ Environment reset")
        
        # Create observation and get action
        observation = create_sample_observation()
        action = await client.select_action(observation)
        print(f"✅ Received action with shape: {action.shape}")
        
        # Get compression stats
        stats = client.get_compression_stats(observation)
        print(f"📊 Compression: {stats['compression_ratio']:.1f}x "
              f"({stats['raw_size_mb']:.1f}MB → {stats['compressed_size_mb']:.1f}MB)")
        
    except LeRobotClientError as e:
        print(f"❌ Client error: {e}")
    finally:
        # Always disconnect
        await client.disconnect()

async def example_2_context_manager():
    """Example 2: Using context manager (recommended)."""
    print("\nExample 2: Context Manager Usage")
    print("-" * 35)
    
    try:
        # Use as context manager - automatically connects and disconnects
        async with LeRobotClient() as client:
            
            # Test ping
            if not await client.ping():
                print("❌ Server not responding")
                return
            
            # Reset and get multiple actions
            await client.reset()
            
            for i in range(3):
                observation = create_sample_observation()
                action = await client.select_action(observation)
                print(f"✅ Action {i+1}: shape={action.shape}, "
                      f"range=[{action.min():.3f}, {action.max():.3f}]")
                
                # Small delay between actions
                await asyncio.sleep(0.1)
                
    except LeRobotClientError as e:
        print(f"❌ Client error: {e}")

async def example_3_custom_config():
    """Example 3: Custom configuration."""
    print("\nExample 3: Custom Configuration")
    print("-" * 33)
    
    # Custom client configuration
    client = LeRobotClient(
        host="localhost",
        port=8765,
        compression_level=9,  # Maximum compression
        timeout=10.0,         # 10 second timeout
        max_message_size=200 * 1024 * 1024  # 200MB max message size
    )
    
    try:
        await client.connect()
        
        observation = create_sample_observation()
        
        # Check compression effectiveness
        stats = client.get_compression_stats(observation)
        print(f"📊 High compression stats:")
        print(f"   Raw size: {stats['raw_size_mb']:.1f}MB")
        print(f"   Compressed: {stats['compressed_size_mb']:.1f}MB") 
        print(f"   Ratio: {stats['compression_ratio']:.1f}x")
        print(f"   Level: {stats['compression_level']}")
        
        # Get action
        action = await client.select_action(observation)
        print(f"✅ Action received: {action.shape}")
        
    except LeRobotClientError as e:
        print(f"❌ Error: {e}")
    finally:
        await client.disconnect()

async def example_4_convenience_function():
    """Example 4: Using convenience function."""
    print("\nExample 4: Convenience Function")
    print("-" * 32)
    
    try:
        # Create and connect in one step
        client = await create_client_and_connect(
            host="localhost", 
            port=8765,
            compression_level=6
        )
        
        await client.reset()
        
        observation = create_sample_observation()
        action = await client.select_action(observation)
        print(f"✅ Quick action: {action.shape}")
        
        await client.disconnect()
        
    except LeRobotClientError as e:
        print(f"❌ Error: {e}")

async def example_5_error_handling():
    """Example 5: Comprehensive error handling."""
    print("\nExample 5: Error Handling")
    print("-" * 26)
    
    # Try connecting to non-existent server
    client = LeRobotClient(host="localhost", port=9999)  # Wrong port
    
    try:
        await client.connect()
    except LeRobotClientError as e:
        print(f"✅ Expected error caught: {e}")
    
    # Try using client without connecting
    client2 = LeRobotClient()
    try:
        await client2.ping()
    except LeRobotClientError as e:
        print(f"✅ Connection error caught: {e}")
    
    # Try with invalid observation
    async with LeRobotClient() as client:
        try:
            # This should work if server is running
            await client.ping()
            
            # Invalid observation
            invalid_obs = {"invalid": "data"}
            await client.select_action(invalid_obs)
            
        except LeRobotClientError as e:
            print(f"✅ Invalid observation error: {e}")
        except Exception:
            print("⚠️  Server not running - skipping invalid observation test")

class MyRobotController:
    """Example 6: Integration in a class."""
    
    def __init__(self):
        self.client = None
        self.connected = False
    
    async def start(self):
        """Start the robot controller."""
        self.client = LeRobotClient()
        await self.client.connect()
        await self.client.reset()
        self.connected = True
        print("🤖 Robot controller started")
    
    async def step(self, observation):
        """Take one step with the robot."""
        if not self.connected:
            raise RuntimeError("Controller not started")
        
        action = await self.client.select_action(observation)
        return action
    
    async def stop(self):
        """Stop the robot controller."""
        if self.client:
            await self.client.disconnect()
        self.connected = False
        print("🤖 Robot controller stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

async def example_6_class_integration():
    """Example 6: Integration in a custom class."""
    print("\nExample 6: Class Integration")
    print("-" * 28)
    
    try:
        async with MyRobotController() as robot:
            # Use the robot
            for step in range(3):
                observation = create_sample_observation()
                action = await robot.step(observation)
                print(f"🤖 Step {step+1}: action shape {action.shape}")
                
    except LeRobotClientError as e:
        print(f"❌ Robot error: {e}")
    except Exception as e:
        print(f"⚠️  Skipping robot test: {e}")

async def main():
    """Run all examples."""
    print("🧪 LeRobot Client API Examples")
    print("=" * 40)
    print("Make sure your LeRobot server is running on localhost:8765")
    print()
    
    # Run examples
    await example_1_basic_usage()
    await example_2_context_manager()
    await example_3_custom_config()
    await example_4_convenience_function()
    # await example_5_error_handling()
    # await example_6_class_integration()
    
    print("\n🎉 All examples completed!")
    print("\n💡 Recommended usage pattern:")
    print("   async with LeRobotClient() as client:")
    print("       await client.reset()")
    print("       action = await client.select_action(observation)")

if __name__ == "__main__":
    asyncio.run(main())