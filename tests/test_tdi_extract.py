import pytest
import numpy as np
from copy import deepcopy

from eregion.tasks.tdi import TDIExtractPTC
from eregion.datamodels.image import DetImage, CCDOutput, ImageBundle

from scipy.stats import spearmanr
import matplotlib.pyplot as plt

@pytest.fixture(scope="module")
def example_DetImages_TDIPTC():

    siglevels = np.linspace(0,500, num=200)
    rng = np.random.default_rng(seed=0xDEADBEEF)

    img1 = np.zeros((200, 100))
    img2 = np.zeros((200, 100))

    FPNMap = rng.normal(loc=1.0, scale=0.03, size=(200,100))
    
    for ind, sl in enumerate(siglevels):
        slc1 = rng.normal(loc=sl, scale=np.sqrt(sl), size=100)
        slc2 = rng.normal(loc=sl, scale=np.sqrt(sl), size=100)

        img1[ind] = slc1 * FPNMap[ind]
        img2[ind] = slc2 * FPNMap[ind]


    op = CCDOutput(id="opFU", ext_id=0, ext_slice=[slice(0,200), slice(0,100)],
                   output_slice=[slice(0,200), slice(0,100)],
                   parallel_prescan= slice(0,0),
                   parallel_overscan=slice(200,200),
                   serial_overscan=slice(100,100),
                   serial_prescan=slice(0,0),
                   parallel_axis="y"
                   )

    op2 = deepcopy(op)

    dim1 = DetImage(data=img1, output_objects={"opFU" : op}, meta={"name" : "det1"})
    dim1.add_output(op)
    dim2 = DetImage(data=img2, output_objects={"opFU" : op}, metad={"name": "det1"})
    dim2.add_output(op2)

    imby = ImageBundle([dim1, dim2])
    
    return [op, op2], imby


def test_TDI_extract_single_output(example_DetImages_TDIPTC):
    ops, imby = example_DetImages_TDIPTC
    
    extractor = TDIExtractPTC(groupby_keys=["det_id"])
    stats = extractor.tdi_stats(ops[0])

    #check correct amount of data
    assert len(stats["mean"]) == 200

    # could do a bunch of stats checks here but those probably should live
    # in upstream test suites so not going to. Check the 1/sqrt(2) division later tho

    #check whether exposure time vs mean is in the right order
    #(not exhaustive, depends on config too I guess)

    sr = spearmanr(stats["exptime"], stats["mean"])
    assert round(sr.statistic, 2) == 1.00


def test_TDI_extract_diffims(example_DetImages_TDIPTC):
    ops, imby = example_DetImages_TDIPTC

    extractor = TDIExtractPTC(groupby_keys=["det_id"])

    for res in extractor.lazy_run(imby):
        print(res.ptc_table.columns)
        print(res.ptc_table["std"])
        
        

    
    

    
