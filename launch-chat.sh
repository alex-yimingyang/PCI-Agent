#!/bin/sh
set -eu

agent_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
cd "$agent_dir"
# Encode the path as a TOML-compatible string, including spaces and quotes.
launch_script=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$agent_dir/launch-mcp.sh")
if [ "$#" -eq 0 ]; then
    set -- "Assess one pavement Sample Unit. Ask for source images, SU boundary, detector address and physical scale. Assemble and crop the SU, identify and mask out shoulders and background from RGB, then detect distress and grade each instance."
fi
exec codex -c 'mcp_servers.road_pci.command="/bin/sh"' \
    -c "mcp_servers.road_pci.args=[$launch_script]" \
    -c 'mcp_servers.road_pci.env_vars=["PCI_AGENT_PYTHON","PCI_AGENT_RUNS","PCI_IMAGE_PYTHON","PCI_DETECTOR_PYTHON"]' \
    -c 'mcp_servers.road_pci.startup_timeout_sec=20' \
    -c 'mcp_servers.road_pci.tool_timeout_sec=620' "$@"
