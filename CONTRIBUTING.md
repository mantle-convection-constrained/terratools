# Contributing to TerraTools

## Getting started with git and GitHub
GitHub provides a helpful
guide on the process of contributing to an open-source project
[here](https://opensource.guide/how-to-contribute/).

## Bug reports
It is a great help to the community if you report any bugs that you
may find. We keep track of all open issues related to TerraTools
[here](https://github.com/mantle-convection-constrained/terratools/issues). 

Please follow these simple instructions before opening a new bug report:

- Do a quick search in the list of open and closed issues for a duplicate of
  your issue.
- If you did not find an answer, open a new
  [issue](https://github.com/mantle-convection-constrained/terratools/issues/new) and explain your
  problem in as much detail as possible.
- Attach as much as possible of the following information to your issue:
  - a minimal set of instructions that reproduce the issue,
  - the error message you saw on your screen,
  - any information that helps us understand why you think this is a bug, and
    how to reproduce it.

## Contributing code
To make a change to TerraTools you should:
- Create a
[fork](https://guides.github.com/activities/forking/#fork) (through GitHub) of
the code base.
- Create a separate
[branch](https://guides.github.com/introduction/flow/) (sometimes called a
feature branch) on which you do your modifications.
- Indent your code correctly by executing `./contrib/utilities/indent`
from the main directory.
- Propose that your branch be merged into the terratools
code by opening a [pull request](https://guides.github.com/introduction/flow/).
This will give others a chance to review your code.
- For new functions, write a useful [docstring](https://docs.python.org/3/tutorial/controlflow.html#documentation-strings) in the [Sphinx style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).
- For new features (new functions, modules or new options to existing functions), add documentation for them (in `docs`).
- Take note of [semantic versioning](https://semver.org).  This means that if you want to change the way that terratools works, a new major (X+1.0.0) version of the software will need to be released.  Mention this in your pull request.  Any new features which are backward-compatible require a new minor (X.Y+1.0) version.

### Code conventions
General conventions:
- Module global variables should be prefixed with a `_` unless they are meant to be used by users.  E.g., `plot._CARTOPY_INSTALLED` is not meant to be part of the public interface, but `terra_model.VALUE_TYPE` can be used externally.
- Class attributes should be prefixed with a `_` unless they are meant to be accessed when using an instance of the class directly.  E.g., in `terra_model.TerraModel`, note that `_lon` and `_lat` are attributes but they are not meant to be accessed directly; instead the `get_lateral_points` method is provided.
- Functions which calculate and return something from their arguments should be named by the thing they return.  E.g., `geographic.azimuth` **not** `geographic.calculate_azimuth`.

Certain modules and classes follow specific conventions on names of functions and variables:
- `terra_model.TerraModel`:
  - Functions which return an object which makes up a `TerraModel` instance which can be edited to update the model are prefixed with `get_`.  E.g., `terra_model.TerraModel.get_radii` returns a vector of radii which can be changed, which will change the model.
  - On the other hand, functions which return something which cannot be used to update the model are not prefixed with `get_`.  E.g., `terra_model.TerraModel.evaluate` returns values from the model at a certain point in space.

## License
TerraTools is published under the [MIT](LICENSE); while you
will retain copyright on your contributions, all changes to the code
must be provided under this common license.
