#!/bin/bash

# Download latest models from https://github.com/aivision/assets/releases
# Example usage: bash aivision/data/scripts/download_weights.sh
# parent
# └── weights
#     ├── aiviv8n.pt  ← downloads here
#     ├── aiviv8s.pt
#     └── ...

python << EOF
from aivision.utils.downloads import attempt_download_asset

assets = [f"aiviv8{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose")]
for x in assets:
    attempt_download_asset(f"weights/{x}")
EOF
