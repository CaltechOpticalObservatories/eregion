from eregion.io.fits_artifact import FITSArtifact
from eregion.io.binary_artifact import BinaryArtifact


print(BinaryArtifact._type_registry)


PTH = "D:\\testdata\\DTU_dettest2\\PSR_results\\PTC-comb\\PTC_fits_comb.fits"

afact = FITSArtifact.load(PTH)



