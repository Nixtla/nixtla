#!/usr/bin/env bash
# __all__ gets written by nbdev and formatting everytime is annoying
# we just skip checking __all__ altogether
for file in mlforecast/[!_]*.py; do
  START=$(grep -n "Cell" $file | head -n 1 | cut -d : -f 1)
  tail -n +$START $file | black --check -S -
done
