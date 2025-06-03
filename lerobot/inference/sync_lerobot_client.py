#!/usr/bin/env python
"""
Synchronous wrapper for LeRobot WebSocket Client.

This provides a synchronous interface to the async LeRobotClient,
allowing you to use it without async/await syntax.
"""

import asyncio
import numpy as np
import threading
from typing import Dict, Any, Optional
from collections import OrderedDict
import logging

# Import the async client (make sure lerobot_client.py is in the same directory)
from lerobot.inference.lerobot_client import LeRobotClient, LeRobotClientError

class SyncLeRobotClient:
    """
    Synchronous wrapper for LeRobotClient.
    
    This class provides a synchronous interface to the async LeRobotClient,
    handling all the asyncio operations internally.
    
    Usage:
        client = SyncLeRobotClient()
        client.connect()
        
        client.reset()
        action = client.select_action(observation)
        
        client.disconnect()
        
    Or use as context manager:
        with SyncLeRobotClient() as client:
            client.reset()
            action = client.select_action(observation)
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765, 
                 compression_level: int = 6, max_message_size: int = 100 * 1024 * 1024,
                 timeout: float = 30.0):
        """
        Initialize synchronous LeRobot client.
        
        Args:
            host: Server hostname (default: localhost)
            port: Server port (default: 8765) 
            compression_level: Gzip compression level 1-9 (default: 6)
            max_message_size: Maximum WebSocket message size in bytes (default: 100MB)
            timeout: Timeout for operations in seconds (default: 30)
        """
        self._async_client = LeRobotClient(
            host=host, port=port, compression_level=compression_level,
            max_message_size=max_message_size, timeout=timeout
        )
        self._loop = None
        self._thread = None
        self._connected = False
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to run in a new thread
                return self._run_in_thread(coro)
            else:
                # Loop exists but not running, we can use it
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create a new one
            return asyncio.run(coro)
    
    def _run_in_thread(self, coro):
        """Run coroutine in a separate thread with its own event loop."""
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(coro)
                loop.close()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        return self._connected
    
    def connect(self) -> None:
        """
        Connect to the LeRobot WebSocket server.
        
        Raises:
            LeRobotClientError: If connection fails
        """
        if self._connected:
            self.logger.warning("Already connected to server")
            return
        
        self._run_async(self._async_client.connect())
        self._connected = True
        self.logger.info("✅ Connected to LeRobot server")
    
    def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self._connected:
            self._run_async(self._async_client.disconnect())
            self._connected = False
            self.logger.info("Disconnected from server")
    
    def ping(self) -> bool:
        """
        Send a ping to test server connectivity.
        
        Returns:
            True if ping successful, False otherwise
        """
        if not self._connected:
            return False
        return self._run_async(self._async_client.ping())
    
    def reset(self) -> bool:
        """
        Reset the environment on the server.
        
        Returns:
            True if reset successful
            
        Raises:
            LeRobotClientError: If reset fails
        """
        if not self._connected:
            raise LeRobotClientError("Not connected to server. Call connect() first.")
        
        return self._run_async(self._async_client.reset())
    
    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Request action selection from the server given an observation.
        
        Args:
            observation: Observation dictionary containing numpy arrays
                Expected structure:
                - 'agent_pos': numpy array of shape (1, 14)
                - 'pixels': dict with 'top' key containing array of shape (1, 480, 640, 3)
        
        Returns:
            Action numpy array from the server
            
        Raises:
            LeRobotClientError: If action selection fails
        """
        if not self._connected:
            raise LeRobotClientError("Not connected to server. Call connect() first.")
        
        return self._run_async(self._async_client.select_action(observation))
    
    def get_compression_stats(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get compression statistics for an observation.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Dictionary with compression statistics
        """
        return self._async_client.get_compression_stats(observation)

def create_sync_client_and_connect(host: str = "localhost", port: int = 8765, **kwargs) -> SyncLeRobotClient:
    """
    Create and connect a synchronous LeRobot client in one step.
    
    Args:
        host: Server hostname
        port: Server port
        **kwargs: Additional arguments for SyncLeRobotClient constructor
        
    Returns:
        Connected SyncLeRobotClient instance
    """
    client = SyncLeRobotClient(host=host, port=port, **kwargs)
    client.connect()
    return client

# Example usage
if __name__ == "__main__":
    # Create sample observation
    def create_sample_observation():
        observation = OrderedDict()
        observation['agent_pos'] = np.random.randn(1, 14).astype(np.float32)
        observation['pixels'] = {
            'top': np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
        }
        return observation
    
    print("🧪 Testing Synchronous LeRobot Client")
    print("=" * 40)
    
    try:
        # Method 1: Manual connection management
        print("Method 1: Manual connection")
        client = SyncLeRobotClient()
        client.connect()
        
        # Test ping
        if client.ping():
            print("✅ Server is responding")
        
        # Reset and get action
        client.reset()
        print("✅ Environment reset")
        
        observation = create_sample_observation()
        action = client.select_action(observation)
        print(f"✅ Received action: shape={action.shape}")
        
        client.disconnect()
        print("✅ Disconnected")
        
        print("\nMethod 2: Context manager")
        # Method 2: Context manager (recommended)
        with SyncLeRobotClient() as client:
            client.reset()
            
            for i in range(3):
                observation = create_sample_observation()
                action = client.select_action(observation)
                print(f"✅ Step {i+1}: action shape {action.shape}")
        
        print("\n🎉 All tests completed successfully!")
        
    except LeRobotClientError as e:
        print(f"❌ Client error: {e}")
        print("Make sure your LeRobot server is running on localhost:8765")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")