# Auckland Housing Market Dataset
Geocoding building consents and residential transaction data into static maps  
Enriching geospatial datasets with additional information

### Installation

`sudo pip3 install -r requirements.txt`

### Running

`jupyter-notebook`

### Data sources

- 2013 Census meshblock household/individual information (2013-mb-dataset-Total-New-Zealand-Household.csv, 2013-mb-dataset-Total-New-Zealand-Individual-Part-1.csv): https://www.stats.govt.nz/information-releases/2013-census-meshblock-dataset
- 2018 Census electoral population (Meshblock 2020) data (2018-census-electoral-population-meshblock-2020-data.csv): https://datafinder.stats.govt.nz/document/22604-2018-census-electoral-population-meshblock-2020-data/
- 2018 Census dwellings by SA2 + Median Household Income + Mean Household Income (2018_census_dwellings_by_SA2.xlsx): custom purchase from Stats NZ
- Rapid transit stops (Northern busway, rail or ferry terminal) (Geocoordinates_Direct_Transit_stops_AKL.xlsx): provided by Ryan
- Total population from 2018 census by SA2/SA2 (Individual_part1_totalNZ-wide_format_updated_16-7-20.csv): https://www.stats.govt.nz/information-releases/statistical-area-1-dataset-for-2018-census-updated-march-2020
- NZ State Highway on-ramps and off-ramps (kx-nz-state-highway-on-ramps-off-ramps-SHP.zip): https://nzta.koordinates.com/layer/1332-nz-state-highway-on-ramps-off-ramps/
- NZ Coastline Mean High water (lds-nz-coastline-mean-high-water-FGDB.zip): https://data.linz.govt.nz/layer/105085-nz-coastline-mean-high-water/
- NZ Coastlines and Islands polygons (lds-nz-coastlines-and-islands-polygons-topo-150k-FGDB.zip): https://data.linz.govt.nz/layer/51153-nz-coastlines-and-islands-polygons-topo-150k/
- NZ primary parcels (lds-nz-primary-parcels-FGDB.zip): https://data.linz.govt.nz/layer/50772-nz-primary-parcels/
- NZ railway centrelines (lds-nz-railway-centrelines-topo-150k-SHP.zip): https://data.linz.govt.nz/layer/50319-nz-railway-centrelines-topo-150k/
- NZ road centrelines (lds-nz-road-centrelines-topo-150k-FGDB.zip): https://data.linz.govt.nz/layer/50329-nz-road-centrelines-topo-150k/
- Auckland Unitary Plan (MASTER_UP_BaseZone_SHP.zip): https://catalogue.data.govt.nz/dataset/unitary-plan-shapefile
- Modified Community Boards (Modified_Community_Boards_SHP.zip): provided by Ryan
- Area Unit 2013 (statsnzarea-unit-2013-FGDB.zip): https://datafinder.stats.govt.nz/layer/25743-area-unit-2013/
- Meshblock 2018 higher geographies (statsnzmeshblock-higher-geographies-2018-generalised-FGDB.zip): https://datafinder.stats.govt.nz/layer/92200-meshblock-higher-geographies-2018-generalised/
- Population by Meshblock 2013 (statsnzpopulation-by-meshblock-2013-census-FGDB.zip): https://datafinder.stats.govt.nz/layer/8437-population-by-meshblock-2013-census/
- SA2 2018 Higher Geographies (statsnzstatistical-area-2-higher-geographies-2018-clipped-generalis-FGDB.zip): https://datafinder.stats.govt.nz/layer/95065-statistical-area-2-higher-geographies-2018-generalised/

#### Restricted data sources (restricted folder)

Not publicly available.

- CSV file containing dwellings that were sold at least once between 1990 and 2020 in the Auckland Region (QPIDs_Auckland_1990_2020.csv): https://www.library.auckland.ac.nz/databases/record/?record=ResProSalSta

### Notebooks

These Python Jupyter (.ipynb) notebooks have been converted to HTML with `jupyter-nbconvert --to html phase1.ipynb` for ease of viewing results

- [phase0](https://uoa-eresearch.github.io/auckland-housing/phase1)
- [phase1](https://uoa-eresearch.github.io/auckland-housing/phase1)
- [phase2](https://uoa-eresearch.github.io/auckland-housing/phase2)
- [phase3](https://uoa-eresearch.github.io/auckland-housing/phase3)
