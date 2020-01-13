from pathlib import Path
import logging
from erebos import prep


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelno)s %(message)s", level="INFO")
    calipso_dir = Path("/storage/projects/goes_alg/calipso/west/1km_cloud/")
    goes_dir = Path("/storage/projects/goes_alg/goes_data/west/combined/")
    save_dir = Path("/storage/projects/goes_alg/combined/west/daytime")
    save_dir.mkdir(parents=True, exist_ok=True)
    goes_glob = "*.nc"
    calipso_glob = "*D_Sub*.hdf"
    prep.combine_calipso_goes_files(
        calipso_dir, goes_dir, save_dir, goes_glob, calipso_glob
    )
