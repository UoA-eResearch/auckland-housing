# The impact of upzoning on housing markets
Geocoding building consents and residential transaction data into static maps  
Enriching geospatial datasets with additional information

### Installation

`sudo pip3 install -r requirements.txt`

### Running

`jupyter-notebook`

### Data sources

- Auckland Unitary Plan (MASTER_UP_BaseZone_SHP.zip): https://catalogue.data.govt.nz/dataset/unitary-plan-shapefile
- 2013 Census meshblock household information (2013-mb-dataset-Total-New-Zealand-Household.csv): https://www.stats.govt.nz/information-releases/2013-census-meshblock-dataset
- NZ Coastlines and Islands polygons (lds-nz-coastlines-and-islands-polygons-topo-150k-FGDB.zip): https://data.linz.govt.nz/layer/51153-nz-coastlines-and-islands-polygons-topo-150k/
- NZ primary parcels (lds-nz-primary-parcels-FGDB.zip): https://data.linz.govt.nz/layer/50772-nz-primary-parcels/
- Meshblock 2018 higher geographies (statsnzmeshblock-higher-geographies-2018-generalised-FGDB.zip): https://datafinder.stats.govt.nz/layer/92200-meshblock-higher-geographies-2018-generalised/
- Population by Meshblock 2013 (statsnzpopulation-by-meshblock-2013-census-FGDB.zip): https://datafinder.stats.govt.nz/layer/8437-population-by-meshblock-2013-census/

### Notebooks

These Python Jupyter (.ipynb) notebooks have been converted to HTML with `jupyter-nbconvert --to html phase1.ipynb` for ease of viewing results

- [phase1](https://uoa-eresearch.github.io/house-upzone/phase1)
- [phase2](https://uoa-eresearch.github.io/house-upzone/phase2)