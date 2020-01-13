import logging
from pathlib import Path


from erebos import prep
from erebos.adapters.goes import GOESFilename


logging.basicConfig(format="%(asctime)s %(levelno)s %(message)s", level="INFO")

calipso_dir = Path("/storage/projects/goes_alg/calipso/west/1km_cloud/")
goes_dir = Path("/storage/projects/goes_alg/goes_data/west/CMIP/")
xml_dir = Path("/storage/projects/goes_alg/goes_data/west/xml/")
product_names = (
    [("ABI-L2-MCMIPC", None)]
    + [("ABI-L2-CMIPC", band) for band in range(1, 17)]
    + [("ABI-L1b-RadC", band) for band in range(1, 17)]
)
prep.download_corresponding_goes_files(
    calipso_dir,
    goes_dir,
    bucket_name="noaa-goes16",
    product_names_bands=product_names,
    checkpoint=True,
    cglob="*D_Sub*.hdf",
)
xml_dir.mkdir(parents=True, exist_ok=True)
for gfile in goes_dir.glob("*CMIPC*C01*.nc"):
    gcf = GOESFilename.from_path(gfile)
    prep.create_class_search_xml(gcf, xml_dir)
for xml_file in xml_dir.glob("*.xml"):
    if (xml_file.parent / "retrieved" / xml_file.name).exists() or (
        xml_file.parent / "processing" / xml_file.name
    ).exists():
        xml_file.unlink()
