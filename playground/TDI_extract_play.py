import os
from importlib import resources

from eregion.tasks import ImageCreator
import logging
from typing import Any
import eregion.configs.detectors
from eregion.tasks.custom import (
    load_image_fits_DEIMOS,
    guess_image_type_from_filename_DEIMOS,
)

from eregion.tasks.tdi import TDIExtractPTC


from eregion.configs.identifiers.deimos import DEIMOS_imtype_header_identify


configdir = resources.files(eregion.configs.detectors)
deimos_cfg = configdir.joinpath("deimos_sci.yaml")

BASE_DIR = "/dettest_data/DTU_dettest"

DATA_DIR = "DTU_fullfp_bringup/TDI/IPhisweep/SCI/20260915-104406"
# DATA_DIR = "/dettest_data/DTU_dettest/DTU_fullfp_bringup/PTC/SCI/20260721-174626"

bias_fls = os.path.join(BASE_DIR, DATA_DIR, "*_TDI900_VIPhi8.0*")
# bias_fls = os.path.join(DATA_DIR, "*_bias_*.fits")

with resources.as_file(deimos_cfg) as dcfg:
    cfgdat = dcfg.read_text()


crtr = ImageCreator(detector_config=cfgdat, n_jobs=1)
crtr.logger.setLevel(logging.INFO)


extractor = TDIExtractPTC(groupby_keys = ["det_id", "IPHI"])


i: int = 0
for frame in crtr.lazy_run(
    input_source=bias_fls,
    identifier_func=DEIMOS_imtype_header_identify,
    fileloader_func=load_image_fits_DEIMOS,
    data_on_demand=False,
):

    i+=1
    ptc_result = extractor.lazy_run(frame.data)

    if i > 1:
        break



                          


    
