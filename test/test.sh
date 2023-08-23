#!/bin/bash

echo "poetry run python pointing.py dfits/dfits_20171115052750.fits.gz ../yaml/pointing_params.yaml"
time poetry run python pointing.py dfits/dfits_20171115052750.fits.gz  ../yaml/pointing_params.yaml
echo "END"

echo "poetry run python beammap.py dfits/dfits_20171115052750.fits.gz ../yaml/beammap_params.yaml"
#time poetry run python beammap.py dfits/dfits_20171111110002.fits.gz ../yaml/beammap_params.yaml Flux_list_Uranus_FWHM22.txt
time poetry run python beammap.py dfits/dfits_20171111110002.fits.gz ../yaml/beammap_params.yaml fluxList.txt
echo "END"



