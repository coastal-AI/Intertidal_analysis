# data_v4/: the data folder of the code

The code keeps its downloaded and intermediate data under `data_v4/`. Git
ignores the folder (`.gitignore`); besides this README, one data file is in
the repository:

| file | contents | read by |
|---|---|---|
| `gauges/ioc_stations.json` | the station list of the IOC Sea Level Station Monitoring Facility, as downloaded from it: 2,389 tide gauges worldwide, each with its code, position, country, operator and sensor metadata | `pyintertidal.boundary.nearest_gauges` (through `make_boundary`), to find the gauges nearest to a site's mouth; the validation v2 harness (`experiments/benchmark/config.py`, `IOC_STATIONS`) |

Do not edit `ioc_stations.json`: validation v2 compares its sha256 between
machines, so an edited copy would differ from the one the runs use.

What the code writes here on your machine (not in the repository):

* `gauges/`: the gauge records that `make_boundary` downloads, and
  `gauges/ioc_2017-2025/`, the separate gauge cache of validation v2
  (`python -X utf8 -m experiments.validation_v2 fetch ioc`);
* `truth/`: survey data used as truth, such as the Vaklodingen sheets
  fetched by `experiments/v0_vaklodingen_fetch.py` and the PNOA LiDAR tile
  read by `experiments/v10_validation_report.py`;
* `store/`: the stores of the development phase, whose sha256 are recorded
  in [`../sealed/registry.jsonl`](../sealed/README.md).
