#!/usr/bin/env bash
nohup pushd web; sudo python -m SimpleHTTPServer 80; popd > webserver.out &
