#!/bin/bash
docker run -it --restart=always -e PBF_URL=https://download.geofabrik.de/australia-oceania/new-zealand-latest.osm.pbf  -e REPLICATION_URL=https://download.geofabrik.de/australia-oceania/new-zealand-updates/  -p 5001:8080   --name nominatim mediagis/nominatim:3.7
