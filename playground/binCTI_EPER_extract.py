from eregion.tasks import ImageCreator
from eregion.tasks.bincti import CTIEPERExtractor

import os


DATA_DIR: str = "/dettest_data/DTU_dettest/DTU_fullfp_bringup/bintest/SCI/20260722-171930"
CONFIG: str = "/home/danw/Software/eregion/src/eregion/configs/detectors/deimos_sci.yaml"

SER_GLOB: str = "DTU_*TDCser0*.fits"
LLEL_GLOB: str = "DTU_*TDCllel*.fits"

imgentask = ImageCreator(detector_config=CONFIG)



imgen_ser = imgentask.lazy_run(os.path.join(DATA_DIR,SER_GLOB))
imbundle_ser = next(imgen_ser).data

ser_extractor = CTIEPERExtractor(



imgen_llel = imgentask.lazy_run(os.path.join(DATA_DIR,LLEL_GLOB))
imbundle_llel = next(imgen_llel).data


