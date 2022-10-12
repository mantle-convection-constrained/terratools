"""Copy the source code and selected examples into the doc directory."""

import os
import shutil

# List of the examples to copy
examples = ["example_attenuation.py"]


# The rest of this script does all the copying
path = "docs/reference_examples/"

if os.path.exists(path):
    shutil.rmtree(path)

os.makedirs(f"{path}")

for ex in examples:
    shutil.copyfile(
        f"examples/{ex}",
        f"docs/reference_examples/{ex}",
    )
