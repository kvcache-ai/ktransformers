# Makefile
## Target
### flake_find:
```bash
make flake_find
```
find all the python files under ./ktransformers dir and find the Error, Warning, Fatal... (their codes) into a list that are not consistent with the pep8 standard. For now we have get all this list in the .flake8 file's extend-ignore section in order to let flakes8 ignore them temporarily.(we may improve them in the future)
### format:
```bash
make format
```
we use black to format all the python files under ./ktransformers dir. It obeys the pep8 standard 
but we modify the line length to 120 by add 
```toml
[tool.black]
line-length = 120
preview = true
unstable = true
```
in the pyproject.toml file.

### dev_install:
```bash
make dev_install
```
install the package in the development mode. It means that the package is installed in the editable mode. So if you modify the code, you don't need to reinstall the package. We recommend the developer to use this method to install the package.