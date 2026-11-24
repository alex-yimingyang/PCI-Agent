#!/bin/sh
set -eu

agent_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
cd "$agent_dir"
# Encode the path as a TOML-compatible string, including spaces and quotes.
launch_script=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$agent_dir/launch-mcp.sh")
agent_args=
startup_prompt=
if [ "${1:-}" = "--agent" ]; then
    case "${2:-}" in
        M1|m1|image) route=image ;;
        M2|m2|detect) route=detect ;;
        M3|m3|mask) route=mask ;;
        M4|m4|labeled) route=labeled ;;
        *) echo 'Use --agent M1, M2, M3 or M4.' >&2; exit 2 ;;
    esac
    startup_prompt="Run the $route pavement assessment method. Ask for my inputs and physical scale, then report the PCI interval and condition in English."
    agent_args=",\"--agent\",\"$route\""
    shift 2
    if [ "$#" -eq 0 ]; then
        set -- "$startup_prompt"
    fi
fi
exec codex -c 'mcp_servers.road_pci.command="/bin/sh"' \
    -c "mcp_servers.road_pci.args=[$launch_script$agent_args]" \
    -c 'mcp_servers.road_pci.env_vars=["PCI_AGENT_PYTHON","PCI_AGENT_RUNS","PCI_IMAGE_PYTHON"]' \
    -c 'mcp_servers.road_pci.startup_timeout_sec=20' \
    -c 'mcp_servers.road_pci.tool_timeout_sec=620' "$@"
