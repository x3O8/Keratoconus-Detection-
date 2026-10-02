"""Start the built website and inference API together: python run.py."""
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    if not (Path(__file__).parent / "frontend/dist/index.html").is_file():
        raise SystemExit("Build the frontend first: npm --prefix frontend run build")
    print("Website and inference: http://127.0.0.1:8000", flush=True)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, access_log=False)
