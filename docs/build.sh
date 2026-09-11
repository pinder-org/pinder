#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

rm -f source/pinder.*
cp ../examples/*.ipynb .
trap 'rm -f ./*.ipynb' EXIT
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-core/pinder/
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-data/pinder/
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-eval/pinder/
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-methods/pinder/
cp pinder.rst source/
rm -rf _build/doctrees
rm -rf _build/html
make html

if [[ -n "${1:-}" ]]; then
  open _build/html/index.html
else
  echo "View docs at _build/html/index.html"
fi
