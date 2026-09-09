"""Desktop entry point that reports FastAPI readiness to the Tauri host."""

import argparse
import asyncio
import json

import uvicorn

from backend.server import app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GigaChat Bot's local API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args(argv)


async def serve(host: str, port: int) -> None:
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    task = asyncio.create_task(server.serve())

    while not server.started:
        if task.done():
            await task
            raise RuntimeError("FastAPI sidecar stopped before it became ready")
        await asyncio.sleep(0.01)

    socket = server.servers[0].sockets[0]
    print(json.dumps({"event": "ready", "port": socket.getsockname()[1]}), flush=True)
    await task


def main() -> None:
    args = parse_args()
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
