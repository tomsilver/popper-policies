#!/bin/bash
yapf -i -r --style .style.yapf --exclude '**/third_party' popper_policies
# yapf -i -r --style .style.yapf tests
yapf -i -r --style .style.yapf *.py
docformatter -i -r . --exclude venv popper_policies/third_party
isort .
