"""python bench/prep_cli.py <day> [start_s end_s]   build the neutral dataset of a day."""
import json
import sys

import data

day = sys.argv[1]
window = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
path, meta = data.prepare(day, window)
print(path)
print(json.dumps(meta, indent=1))
