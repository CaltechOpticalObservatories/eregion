from eregion.tasks import ImageCreator
from eregion.tasks.bincti import CTIEPERExtractor
from astropy.stats import sigma_clipped_stats

import os


DATA_DIR: str = '/home/danw/dettest_data/DTU_dettest/DTU_fullfp_bringup/bintest/SCI/20260722-171930'
CONFIG: str = "/home/danw/Software/eregion/src/eregion/configs/detectors/deimos_sci.yaml"


SER_GLOB: str = "DTU_*TDCser0*.fits"
LLEL_GLOB: str = "DTU_*TDCllel*.fits"

imgentask = ImageCreator(detector_config=CONFIG)


imgen_ser = imgentask.lazy_run(os.path.join(DATA_DIR,SER_GLOB))
imbundle_ser = next(imgen_ser).data
ser_dim = imbundle_ser[0]

imgen_llel = imgentask.lazy_run(os.path.join(DATA_DIR,LLEL_GLOB))
imbundle_llel = next(imgen_llel).data


extractor = CTIEPERExtractor(N_bright_lines=0, bright_line_spacing=0,
                             first_bright_line=0, N_before_bright=0,
                             axis="parallel")

stdfun = lambda *a, **kw: 0.0

dat, mask = imbundle_llel[0].outputs["E"].get_image_region()

stats = sigma_clipped_stats(dat, axis=1, sigma=5.0, grow=10, cenfunc="mean",
                            stdfunc="std")


