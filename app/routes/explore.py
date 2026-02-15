"""Flask blueprint to serve the Observable Framework explorer at /explore/."""

import os
from pathlib import Path

from flask import Blueprint, send_from_directory

explore_bp = Blueprint("explore", __name__)

# In the Docker container the built files live at /app/explore_dist/;
# locally during development they're at explore/dist/ relative to the repo root.
_EXPLORE_DIR = os.environ.get(
    "EXPLORE_DIST_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "explore" / "dist"),
)


@explore_bp.route("/explore/")
@explore_bp.route("/explore/<path:filename>")
def serve_explore(filename="index.html"):
    return send_from_directory(_EXPLORE_DIR, filename)
