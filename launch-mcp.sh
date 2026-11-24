#!/bin/sh
set -eu

agent_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${PCI_AGENT_PYTHON:-}" ]; then
    python_bin=$PCI_AGENT_PYTHON
elif [ -x "$agent_dir/.venv/bin/python" ]; then
    python_bin="$agent_dir/.venv/bin/python"
else
    printf '%s\n' 'PCI Agent requires a Python environment with the dependencies in requirements.txt. See README.md.' >&2
    exit 1
fi

exec "$python_bin" -B "$agent_dir/pci_mcp.py" "$@"
