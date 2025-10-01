"""Entry point for running the Chino chat backend server."""
from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the FastAPI server that powers the chat experience."""

    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()