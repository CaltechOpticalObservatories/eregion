import os
from importlib import resources

from eregion.tasks import ImageCreator
import logging
import eregion.configs.detectors
from eregion.tasks.custom import load_image_fits_DEIMOS, guess_image_type_from_filename_DEIMOS

configdir = resources.files(eregion.configs.detectors)
deimos_cfg = configdir.joinpath("deimos_sci.yaml")

BASE_DIR = "/network/workdata/caltech/DTU_dettest"

DATA_DIR =  "DTU_fullfp_bringup/TDI/IPhisweep/SCI/20260915-104406"
#DATA_DIR = "/dettest_data/DTU_dettest/DTU_fullfp_bringup/PTC/SCI/20260721-174626"

bias_fls = os.path.join(BASE_DIR, DATA_DIR, "*TDIbias900*.fits")
#bias_fls = os.path.join(DATA_DIR, "*_bias_*.fits")

with resources.as_file(deimos_cfg) as dcfg:
    cfgdat = dcfg.read_text()


crtr = ImageCreator(detector_config=cfgdat, n_jobs=1)
crtr.logger.setLevel(logging.INFO)


i: int = 0
for bias in crtr.lazy_run(input_source=bias_fls,
                          identifier_func=guess_image_type_from_filename_DEIMOS,
                          fileloader_func=load_image_fits_DEIMOS,
                          data_on_demand=False):
    print(type(bias.data[0].data))
    print(i)
