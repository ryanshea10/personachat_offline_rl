#!/bin/sh

python download_dta.py
python format_data.py
python filter_data.py