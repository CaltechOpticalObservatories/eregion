import pytest
import numpy as np
from copy import deepcopy

from eregion.tasks.tdi import TDIExtractPTC
from eregion.datamodels.image import DetImage, CCDOutput, ImageBundle

from scipy.stats import spearmanr
import matplotlib.pyplot as plt

@pytest.fixture(scope="module")
def example_DetImages_TDIPTC():
    rng = np.random.default_rng(seed=0xDEADBEEF)
    
    def make_test_images(detname: str = "det_1"):
        siglevels = np.linspace(0,500, num=200)
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

        dim1 = DetImage(data=img1, output_objects={"opFU" : op}, meta={"name" : detname})
        dim2 = DetImage(data=img2, output_objects={"opFU" : op2}, meta={"name": detname})

        imby = ImageBundle([dim1, dim2])
    
        return [op, op2], imby

    return make_test_images


def test_TDI_extract_single_output(example_DetImages_TDIPTC):
    ops, imby = example_DetImages_TDIPTC()
    
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
    ops, imby = example_DetImages_TDIPTC()
    calc_diffim = imby[0].outputs["opFU"].get_image_region()[0].data - imby[1].outputs["opFU"].get_image_region()[0].data
    extractor = TDIExtractPTC(groupby_keys=["det_id"])

    gen = extractor.lazy_run(imby)
    ptcres = next(gen)

    dim_proc = ptcres.diff_images[0].outputs["opFU"].get_image_region()[0].data
    #check the differenced image is calculated properly
    assert np.count_nonzero(np.abs(dim_proc - calc_diffim)) == 0


    #check that the stdev in the diff column is actually divided by sqrt(2)

    tab = ptcres.ptc_table
    sdcheck = (tab["std_0-1"] / tab["std"]).mean()

    assert np.round(sdcheck, 1) == np.round(np.sqrt(2), 1)
    
    
def test_TDI_diffstats_gain(example_DetImages_TDIPTC):
    ops, imby = example_DetImages_TDIPTC()

    extractor = TDIExtractPTC(groupby_keys=["det_id"])
    res = next(extractor.lazy_run(imby))

    #NOTE: in this artificially constructed data, the first gain value will be NaN (it's 0 / 0 essentially)
    gaindat = (res.ptc_table["mean"]  / res.ptc_table["std"]**2).array[1:]

    #gain in the data was constructed to be 1
    gaincheck = np.round(np.mean(gaindat), 0)
    assert gaincheck == 1.0
    

def test_TDI_multigroup(example_DetImages_TDIPTC):
    ops1, imby1 = example_DetImages_TDIPTC("det_1")
    ops2, imby2 = example_DetImages_TDIPTC("det_2")

    imby1.extend(imby2)
    
    extractor = TDIExtractPTC(groupby_keys=["det_id"])

    gen = extractor.lazy_run(imby1)
    res1 = next(gen)

    tab = res1.ptc_table

    tabd1 = tab[tab["det_id"] == "det_1"]
    tabd2 = tab[tab["det_id"] == "det_2"]

    #blunt instrument test, just make sure they're operating on different data
    assert tabd1["mean"].mean() != tabd2["mean"].mean()
    
    
    

    
