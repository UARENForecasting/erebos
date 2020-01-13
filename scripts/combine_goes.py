import logging
from pathlib import Path
from erebos.prep import combine_goes_files


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelno)s %(message)s", level="INFO")
    base_dir = Path("/storage/projects/goes_alg/goes_data/west")
    combine_goes_files(base_dir)
