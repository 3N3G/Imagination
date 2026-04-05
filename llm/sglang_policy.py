#!/usr/bin/env python3
"""
SGLang Policy Wrapper for Craftax Online RL

This module provides a batched policy interface that queries an SGLang server
for LLM-based action selection in the Craftax environment.

Usage:
    # Start SGLang server first:
    # python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 \
    #     --port 30000 --enable-radix-caching --dtype float16
    
    from sglang_policy import SGLangPolicy
    policy = SGLangPolicy(server_url="http://localhost:30000")
    actions = policy.get_actions_batch(observations)
"""

import asyncio
import aiohttp
import re
import time
from typing import List, Dict, Any, Optional
import numpy as np


# System prompt matching llm_play_harnessed.py
SYSTEM_PROMPT = """You are playing Craftax.

Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions and can interact.

The coordinate system is (Row, Column). Everything is relative to your current position.
- Negative Row is UP. Positive Row is DOWN.
- Negative Column is LEFT. Positive Column is RIGHT.
- (0, 0) is your current position.

Actions available: 
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO (interact/mine/attack), 6:SLEEP, 7:PLACE_STONE,
8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT, 11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE,
13:MAKE_IRON_PICKAXE, 14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD, 17:REST,
18:DESCEND, 19:ASCEND, 20:MAKE_DIAMOND_PICKAXE, 21:MAKE_DIAMOND_SWORD, 22:MAKE_IRON_ARMOUR

Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>).
"""

ACTION_PATTERN = re.compile(r'\*\*Action:\*\*\s*(\d+)')


class SGLangPolicy:
    """Batched LLM policy using SGLang server for Craftax."""
    
    def __init__(
        self,
        server_url: str = "http://localhost:30000",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 30.0,
        default_action: int = 0,  # NOOP
    ):
        self.server_url = server_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.default_action = default_action
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        
    def format_prompt(self, observation: str) -> str:
        """Format observation into full prompt for the model."""
        user_content = f"""YOUR CURRENT GAME STATE:
{observation}

You are at (0,0). Output your internal reasoning in a <think> block, then end with: **Action:** <id> (<name>)."""
        
        # Format as chat template (SGLang handles this, but we prepare the structure)
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    def parse_action(self, response: str) -> int:
        """Extract action ID from model response."""
        match = ACTION_PATTERN.search(response)
        if match:
            action_id = int(match.group(1))
            if 0 <= action_id <= 42:
                return action_id
        return self.default_action
    
    async def _query_single(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
    ) -> Dict[str, Any]:
        """Query SGLang server for a single prompt."""
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
            }
        }
        
        try:
            async with session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "text": result.get("text", "")}
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _query_batch_async(
        self,
        observations: List[str],
    ) -> List[int]:
        """Query SGLang server with batched observations asynchronously."""
        prompts = [self.format_prompt(obs) for obs in observations]
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._query_single(session, p) for p in prompts]
            results = await asyncio.gather(*tasks)
        
        actions = []
        for result in results:
            self.total_requests += 1
            if result["success"]:
                actions.append(self.parse_action(result["text"]))
            else:
                self.failed_requests += 1
                actions.append(self.default_action)
        
        return actions
    
    def get_actions_batch(self, observations: List[str]) -> np.ndarray:
        """
        Get actions for a batch of observations.
        
        Args:
            observations: List of text observations from Craftax
            
        Returns:
            numpy array of action IDs
        """
        start_time = time.perf_counter()
        
        # Run async query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            actions = loop.run_until_complete(self._query_batch_async(observations))
        finally:
            loop.close()
        
        self.total_latency += time.perf_counter() - start_time
        
        return np.array(actions, dtype=np.int32)
    
    def get_metrics(self) -> Dict[str, float]:
        """Return policy metrics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "failure_rate": self.failed_requests / max(1, self.total_requests),
            "total_latency_s": self.total_latency,
            "avg_latency_s": self.total_latency / max(1, self.total_requests),
        }
    
    def check_health(self) -> bool:
        """Check if SGLang server is healthy."""
        import requests
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class SGLangBatchPolicy(SGLangPolicy):
    """
    Optimized batch policy that uses SGLang's native batch endpoint.
    
    This version sends all prompts in a single request for better throughput
    when the server supports it.
    """
    
    async def _query_batch_native(
        self,
        observations: List[str],
    ) -> List[int]:
        """Use SGLang's native batch API if available."""
        prompts = [self.format_prompt(obs) for obs in observations]
        
        payload = {
            "text": prompts,  # List of prompts
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.server_url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout * len(prompts))
                ) as response:
                    if response.status == 200:
                        results = await response.json()
                        # Handle both single and batch response formats
                        if isinstance(results, list):
                            texts = [r.get("text", "") for r in results]
                        else:
                            texts = [results.get("text", "")]
                        return [self.parse_action(t) for t in texts]
            except Exception as e:
                print(f"Batch query failed: {e}, falling back to parallel single queries")
        
        # Fallback to parallel single queries
        return await self._query_batch_async(observations)
    
    def get_actions_batch(self, observations: List[str]) -> np.ndarray:
        """Get actions using native batch API."""
        start_time = time.perf_counter()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            actions = loop.run_until_complete(self._query_batch_native(observations))
        finally:
            loop.close()
        
        self.total_latency += time.perf_counter() - start_time
        self.total_requests += len(observations)
        
        return np.array(actions, dtype=np.int32)


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    
    policy = SGLangPolicy(server_url=args.server_url)
    
    if not policy.check_health():
        print(f"ERROR: SGLang server not healthy at {args.server_url}")
        print("Start server with:")
        print("  python -m sglang.launch_server --model-path Qwen/Qwen3-4B-Thinking-2507 --port 30000")
        exit(1)
    
    # Test observations
    test_obs = [
        "Map: 0,1:tree, -1,0:stone\nInventory: Wood:0\nHealth:9\nDirection:right",
        "Map: 1,0:water, 0,-1:cow\nInventory: Wood:5\nHealth:7\nDirection:down",
    ] * (args.batch_size // 2)
    
    print(f"Testing with batch size {len(test_obs)}...")
    start = time.perf_counter()
    actions = policy.get_actions_batch(test_obs)
    elapsed = time.perf_counter() - start
    
    print(f"Actions: {actions}")
    print(f"Time: {elapsed:.2f}s ({len(test_obs)/elapsed:.2f} samples/sec)")
    print(f"Metrics: {policy.get_metrics()}")
