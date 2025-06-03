#!/usr/bin/env python
"""
LeRobot WebSocket Client API

A reusable client for communicating with LeRobot WebSocket servers.
Handles observation serialization, action requests, and connection management.

Example usage:
    from lerobot_client import LeRobotClient
    
    # Create client
    client = LeRobotClient()
    
    # Connect and use
    await client.connect()
    
    # Reset environment
    success = await client.reset()
    
    # Select action
    action = await client.select_action(observation)
    
    # Disconnect
    await client.disconnect()
    
    # Or use as context manager
    async with LeRobotClient() as client:
        await client.reset()
        action = await client.select_action(observation)
"""

import asyncio
import json
import pickle
import base64
import gzip
import numpy as np
import websockets
from typing import Dict, Any, Optional, Union
import logging
from collections import OrderedDict

class LeRobotClientError(Exception):
    """Custom exception for LeRobot client errors."""
    pass

class LeRobotClient:
    """
    WebSocket client for LeRobot servers.
    
    Handles serialization of numpy observations, compression, and communication
    with LeRobot WebSocket servers.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765, 
                 compression_level: int = 6, max_message_size: int = 100 * 1024 * 1024,
                 timeout: float = 30.0):
        """
        Initialize LeRobot client.
        
        Args:
            host: Server hostname (default: localhost)
            port: Server port (default: 8765) 
            compression_level: Gzip compression level 1-9 (default: 6)
            max_message_size: Maximum WebSocket message size in bytes (default: 100MB)
            timeout: Timeout for operations in seconds (default: 30)
        """
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.compression_level = compression_level
        self.max_message_size = max_message_size
        self.timeout = timeout
        
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connected = False
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to server."""
        return self._connected and self._websocket is not None
    
    def _serialize_array(self, array: np.ndarray) -> str:
        """
        Serialize a numpy array to compressed base64 string.
        
        Args:
            array: NumPy array to serialize
            
        Returns:
            Base64-encoded compressed string
        """
        pickled_data = pickle.dumps(array)
        compressed_data = gzip.compress(pickled_data, compresslevel=self.compression_level)
        return base64.b64encode(compressed_data).decode('utf-8')
    
    def _deserialize_array(self, data: str) -> np.ndarray:
        """
        Deserialize a numpy array from compressed base64 string.
        
        Args:
            data: Base64-encoded compressed string
            
        Returns:
            Deserialized NumPy array
        """
        compressed_data = base64.b64decode(data.encode('utf-8'))
        pickled_data = gzip.decompress(compressed_data)
        return pickle.loads(pickled_data)
    
    def _serialize_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize observation dictionary recursively.
        
        Args:
            obs: Observation dictionary containing numpy arrays
            
        Returns:
            Serialized observation dictionary
        """
        result = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                result[key] = self._serialize_array(value)
            elif isinstance(value, dict):
                result[key] = {k: self._serialize_array(v) for k, v in value.items()}
            else:
                result[key] = value
        return result
    
    async def connect(self) -> None:
        """
        Connect to the LeRobot WebSocket server.
        
        Raises:
            LeRobotClientError: If connection fails
        """
        if self._connected:
            self.logger.warning("Already connected to server")
            return
        
        try:
            self.logger.info(f"Connecting to {self.uri}...")
            self._websocket = await asyncio.wait_for(
                websockets.connect(
                    self.uri,
                    max_size=self.max_message_size
                ),
                timeout=self.timeout
            )
            self._connected = True
            self.logger.info("✅ Connected to LeRobot server")
            
        except asyncio.TimeoutError:
            raise LeRobotClientError(f"Connection timeout after {self.timeout}s")
        except websockets.exceptions.ConnectionRefused:
            raise LeRobotClientError(f"Connection refused. Is server running on {self.uri}?")
        except Exception as e:
            raise LeRobotClientError(f"Failed to connect: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self._websocket and self._connected:
            await self._websocket.close()
            self.logger.info("Disconnected from server")
        
        self._websocket = None
        self._connected = False
    
    async def _send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message and wait for response.
        
        Args:
            message: Message dictionary to send
            
        Returns:
            Response dictionary from server
            
        Raises:
            LeRobotClientError: If not connected or communication fails
        """
        if not self.is_connected:
            raise LeRobotClientError("Not connected to server. Call connect() first.")
        
        try:
            # Send message
            message_str = json.dumps(message)
            await asyncio.wait_for(
                self._websocket.send(message_str),
                timeout=self.timeout
            )
            
            # Wait for response
            response_str = await asyncio.wait_for(
                self._websocket.recv(),
                timeout=self.timeout
            )
            
            response = json.loads(response_str)
            
            # Check for server errors
            if response.get("type") == "error":
                raise LeRobotClientError(f"Server error: {response.get('message', 'Unknown error')}")
            
            return response
            
        except asyncio.TimeoutError:
            raise LeRobotClientError(f"Operation timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            raise LeRobotClientError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise LeRobotClientError(f"Communication error: {e}")
    
    async def ping(self) -> bool:
        """
        Send a ping to test server connectivity.
        
        Returns:
            True if ping successful, False otherwise
        """
        try:
            response = await self._send_message({"type": "ping"})
            return response.get("type") == "pong"
        except LeRobotClientError:
            return False
    
    async def reset(self) -> bool:
        """
        Reset the environment on the server.
        
        Returns:
            True if reset successful
            
        Raises:
            LeRobotClientError: If reset fails
        """
        response = await self._send_message({"type": "reset"})
        
        if response.get("type") == "reset_response":
            self.logger.info("Environment reset successful")
            return True
        else:
            raise LeRobotClientError(f"Reset failed: {response}")
    
    async def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
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
        # Validate observation structure
        if not isinstance(observation, dict):
            raise LeRobotClientError("Observation must be a dictionary")
        
        # Serialize observation
        try:
            serialized_obs = self._serialize_observation(observation)
        except Exception as e:
            raise LeRobotClientError(f"Failed to serialize observation: {e}")
        
        # Send action request
        message = {
            "type": "select_action",
            "observation": serialized_obs
        }
        
        # Log data size info
        if self.logger.isEnabledFor(logging.DEBUG):
            message_str = json.dumps(message)
            size_mb = len(message_str.encode('utf-8')) / (1024 * 1024)
            self.logger.debug(f"Sending observation of size: {size_mb:.2f}MB")
        
        response = await self._send_message(message)
        
        if response.get("type") == "action_response":
            try:
                action = self._deserialize_array(response["action"])
                self.logger.debug(f"Received action with shape: {action.shape}")
                return action
            except Exception as e:
                raise LeRobotClientError(f"Failed to deserialize action: {e}")
        else:
            raise LeRobotClientError(f"Unexpected response: {response}")
    
    def get_compression_stats(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get compression statistics for an observation.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Dictionary with compression statistics
        """
        # Calculate raw size
        raw_size = 0
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                raw_size += value.nbytes
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, np.ndarray):
                        raw_size += v.nbytes
        
        # Calculate compressed size
        serialized_obs = self._serialize_observation(observation)
        message = {"type": "select_action", "observation": serialized_obs}
        compressed_size = len(json.dumps(message).encode('utf-8'))
        
        return {
            "raw_size_bytes": raw_size,
            "compressed_size_bytes": compressed_size,
            "raw_size_mb": raw_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": raw_size / compressed_size,
            "compression_level": self.compression_level
        }

# Convenience function for quick usage
async def create_client_and_connect(host: str = "localhost", port: int = 8765, **kwargs) -> LeRobotClient:
    """
    Create and connect a LeRobot client in one step.
    
    Args:
        host: Server hostname
        port: Server port
        **kwargs: Additional arguments for LeRobotClient constructor
        
    Returns:
        Connected LeRobotClient instance
    """
    client = LeRobotClient(host=host, port=port, **kwargs)
    await client.connect()
    return client