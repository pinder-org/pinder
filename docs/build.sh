#!/bin/bash

rm source/pinder.*
cp ../examples/*.ipynb .
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-core/pinder/
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-data/pinder/
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-eval/pinder/
sphinx-apidoc -o source -d 10 -f --implicit-namespaces ../src/pinder-methods/pinder/
cp pinder.rst source/
rm -r _build/doctrees
rm -r _build/html
make html

if [[ -n "$1" ]]; then
  open _build/html/index.html
else
  echo "View docs at _build/html/index.html"
fi
rm *.ipynb
