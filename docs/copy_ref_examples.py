"""Copy the source code and selected examples into the doc directory."""

import os
import shutil
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# List of the examples to copy
examples = [
    "example_evaluate_points",
    "example_cross_section",
    # "example_spherical_harmonics", # FIXME: Not working because of `tests` import
    "example_attenuation",
    # "example_convert", # FIXME: Not working because of `tests` import
]

# The rest of this script does all the copying
path = "docs/reference_examples/"

if os.path.exists(path):
    shutil.rmtree(path)

os.makedirs(f"{path}")

for ex in examples:
    shutil.copyfile(
        f"examples/{ex}.py",
        f"docs/reference_examples/{ex}.py",
    )

    doc_path = f"{ex}.py"
    nav[ex] = doc_path
    full_doc_path = f"reference_examples/{ex}.py"
    mkdocs_gen_files.set_edit_path(full_doc_path, doc_path)

with mkdocs_gen_files.open("reference_examples/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
