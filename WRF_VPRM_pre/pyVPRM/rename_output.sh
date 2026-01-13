#!/bin/bash

# Set default values
default_year=2012
# set domain!!!  -> 
default_domain="d01"

# Use provided arguments or default values if none are given
year=${1:-$default_year}
domain=${2:-$default_domain}

for file in *_${year}.nc; do
    base="${file%_${year}.nc}"
    mv "$file" "${base}_${domain}_${year}.nc"
done