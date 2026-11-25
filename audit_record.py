import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def immutable(path, data):
    """Publish complete stage outputs once. Identical retries are harmless."""
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(encoded)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ValueError('This stage already has different decisions. Start a fresh run.')
    finally:
        temp.unlink()
