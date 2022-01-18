#!/bin/bash
docker run --name osrm -d --restart always -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --max-table-size 50000 --algorithm mld /data/new-zealand-latest.osrm
