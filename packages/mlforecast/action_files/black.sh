#!/usr/bin/env bash
# __all__ gets written by nbdev and formatting everytime is annoying
# we just skip checking __all__ altogether
for FILE in $(find mlforecast -name "[!_]*.py"); do
  echo "Linting $FILE"
  START=$(grep -n "Cell" $FILE | head -n 1 | cut -d : -f 1)
  tail -n +$START $FILE | black --check -S - || exit -1
done
