"""Run a real tools/list against every server in week9/mcp_config.json
and print/save the counts and names. Used once with a 1-server config and
once with a 2-server config to produce genuine before/after numbers —
never hand-typed.

Run:
    python -m week9.discover_tools
"""

import asyncio
import json
import sys

from week9.mcp_client import MCPToolPool, load_config, default_config_path


async def discover(config_path: str | None = None) -> dict:
    cfg = load_config(config_path or default_config_path())
    async with MCPToolPool(cfg) as pool:
        return {
            "servers": list(pool.discovered.keys()),
            "by_server": pool.discovered,
            "total_tools": sum(len(v) for v in pool.discovered.values()),
            "all_tool_names": [name for names in pool.discovered.values() for name in names],
        }


def main():
    result = asyncio.run(discover())
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
