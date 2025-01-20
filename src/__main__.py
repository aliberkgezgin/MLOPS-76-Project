import asyncio

from loguru import logger
import uvicorn
from src.api import app



async def main() -> None:
    """Entrypoint of the application."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

def dummy() -> int:
    """Return 42."""
    return 42


if __name__ == "__main__":
    logger.info("Starting")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Cancelled")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("Exiting")
