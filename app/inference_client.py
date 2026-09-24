"""Temporary inference client abstraction."""

import asyncio


async def run_inference(prompt: str) -> str:
    """Return a mock response until the real inference contract is available."""
    await asyncio.sleep(0.01)
    return f"mock response for: {prompt}"
