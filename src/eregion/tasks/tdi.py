from typing import Optional, Iterable, Generator, Callable, Any
from itertools import combinations
from collections import defaultdict

import xarray
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from copy import deepcopy

from eregion.tasks import LazyTask
from eregion.tasks.ptc import PTCResult
from eregion.datamodels import DetImage, ImageBundle, CCDOutput

from eregion.core.image_stats import do_statistics, STATFUNCS
from eregion.utils.dangerous_magic import pack_argument_helper

_TDISTATS: dict[str, Callable] = {k : STATFUNCS[k] for k in ["mean", "median", "std", "mad"]}

_StatsDictT = dict[str, np.ndarray]

class TDIExtractPTC(LazyTask):
    task_result = PTCResult

    def __init__(self,
                 groupby_keys: Iterable[str],
                 name: Optional[str] = None):

        self.groupby_keys = groupby_keys
            
        newkwargs = pack_argument_helper(selfarg=self)
        super().__init__(**newkwargs)

    def lazy_run(self, images: ImageBundle | Iterable[DetImage]) -> Generator[PTCResult, None, None]:

        if not isinstance(images, ImageBundle):
            images = ImageBundle(images)

        groups = images.groupby(by=self.groupby_keys, sort=False)

        #TODO: implement parallel or not needed? Wait for new Task design and don't bother?
        results = Parallel(n_jobs = self.n_jobs)(
            delayed(self._process_group)(ImageBundle.from_dataframe(g)) for k, g in groups)

        stats, diff_images = [], []
        for stat, difim in results:
            print(f"stat: {stat}")
            print(type(stat))
            stats.append(stat)
            diff_images += difim

        return stats, diff_images
        


    def _process_group(self, images: ImageBundle[DetImage]) -> tuple[dict[Any, _StatsDictT], list[DetImage]]:
        """Process a group of TDI images.
           Perform row-wise statistics on each of the individual images,
           and also row-wise statistics on any diff pairs that are formable

           parameters
           --------

           :param images
             ImageBundle of DetImages to process. All Images in this bundle are assumed to be meaningfully
             differencable (i.e. that a differential image of any two images produces valid PTC statistics).
             It is also assumed that  outputs across different DetImages are named the same.

           returns
           -------

           tuple[dict[Any, dict[str, np.ndarray]], list[DetImage]]

           Tuple containing the statistics per output, keyed on the detector's output ID. The statistics themselves are presented as a dictionary, with suffixes indicating the indices of the flat or differenced image pair
       
        """
        
        base_info = {k : images.list[k][0] for k in self.groupby_keys}

        stats: dict[Any, _StatsDictT] = defaultdict(dict)

        #stats on each image
        for i, img in enumerate(images):
            self.logger.debug("Doing per-output stats on individual images in group, %d of %d", i, len(images))
            for outputid, output in img.outputs.items():
                self.logger.debug("Doing stats on output with id %s" , outputid)
                outstat = base_info | {"output" : outputid}
                #in a TDI curve, exptime is proxied by the parallel axis coordinate
                outstat |= self.tdi_stats(output, suffix=f"_{i}")
                stats[outputid] |= outstat

        #stats on diff images
        if len(images) < 2:
            self.logger.warning("Only one image provided to _process_group. No diffpair stats will be done")
            return stats, []
        
        diff_images = []
        for dpidx0, dpidx1 in combinations(range(len(images)), 2):
            self.logger.debug("doing diffpair calculation on images %d and %d", dpidx0, dpidx1)

            internal_diffim = deepcopy(images[dpidx0])

            #essentially copied unchanged from PTC task. Consider for future refactor
            for opid, output in internal_diffim.outputs.items():
                diffdat = output.data - images[dpidx1].outputs[opid].data
                output.set_data_in_parent(diffdat)
                #don't do sigma clip masking on this for now.
                #datapoints are cheap here, and any kind of vignetting will cause
                #row wise sigma clipping to go wild. Possible future upgrade of TDI analysis
                diff_images.append(internal_diffim)
                diffstats = self.tdi_stats(output, suffix=f"_{dpidx0}-{dpidx1}")
                stats[outputid] |= self.tdi_stats(output, suffix=f"_{dpidx0}-{dpidx1}")

        statdf = self.make_ptc_table(stats)
                
        return statdf, diff_images


    def make_ptc_table(self, stats: Iterable[dict[Any, dict[str, np.ndarray]]]) -> pd.DataFrame:
        """Take the collected statistics output from a TDI run and turn it into a pandas DataFrame

        parameters
        ----------

        :param stats
          statistics dictionary, keyed by the output ID of the detector output amplifier,
          values are individual PTC statistics calculated for that amplifier

        returns
        -------

        pandas DataFrame containing original columns plus some suitable averages:
           ["mean"] - mean of the mean columns of individual images
           ["median"] - mean of the median columns of individual images
           ["std"] - mean of the std columns of  diff pair images divided by sqrt(2) to correct the sampling factor
        """
        outdcts = []
        for opid, statdct in stats.items():
            minidf = pd.DataFrame(statdct)
            minidf["output"] = opid
            outdcts.append(minidf)
        outdf = pd.concat(outdcts, ignore_index=True)

        def sfx_non_diff_filter(basenm: str):
            basesearch = [_ for _ in outdf.columns if basenm in _]
            #bit of a hack: diff pair columns will have a "-" in the name
            narrowsearch = [_ for _ in basesearch if "-" not in _]
            return narrowsearch

        def sfx_diff_filter(basenm: str):
            basesearch = [_ for _ in outdf.columns if basenm in _]
            #bit of a hack: diff pair columns will have a "-" in the name
            narrowsearch = [_ for _ in basesearch if "-" in _]
            return narrowsearch

        outdf["mean"] = outdf[sfx_non_diff_filter("mean")].mean(axis=1)
        outdf["median"] = outdf[sfx_non_diff_filter("median")].mean(axis=1)
        #FUTURE complaint: maybe "std" should be averaging non-diffed, and have "diff_std" for diff
        #anyway, at the moment just have the sqrt(2) on "std" for the same behaviour as PTC task
        outdf["std"] = outdf[sfx_diff_filter("std")].mean(axis=1) / np.sqrt(2)

        return outdf
        
    
    
    def tdi_stats(self, output: CCDOutput, suffix: str = "") -> dict[str, np.ndarray]:
        """Do simple TDI stats on a single output.
        TDI statistics are always done along the serial (because the TDI curve is generated by
        keeping the light on while doing the parallel transfer.

        This function assumes an already overscan and bias subtracted image. Exposure time is here just measured in parallel line times, and so is generated from the geometry of the output

        parameters
        ----------

        :param output: CCDOutput
            CCD Output object to use for calculating the TDI statistics

        :param suffix: str
            suffix to append to all statistics name keywords
        Returns
        -------

        dict [str, np.ndarray]
            Dictionary of PTC statistics calculations. NOTE: each entry in this dictionary is an array of stats for the
            whole PTC (c.f. the eregion.tasks.ptc:PTC task, where the equivalent function returns a dict with a single measurement)

        """

        if not isinstance(output, CCDOutput):
            raise TypeError(f"can only do TDI stats specifically on CCD Ouptuts. The type of output passed was {type(output)}")

        im, masks = output.get_image_region(return_masks=True)
        stats = dict()

        #TODO: consider convenience function in DetImage that combines masks properly
        if masks is not None:
            totmask = xarray.zeros_like(masks[0])
            for mask in masks:
                totmask |= mask
        else:
            totmask = None
        ma = np.ma.MaskedArray(data=im, mask=totmask)
        stats |= do_statistics(ma, axis=output.serial_axint, which=_TDISTATS)

        if totmask is not None:
            stats["n_masked"] = int(np.sum(totmask, axis=output.serial_axint, dtype=np.uint32))
        else:
            axslc = output.image_region[output.parallel_axis]
            llel_len = abs(axslc.stop - axslc.start)
            stats["n_masked"] = np.zeros(llel_len, dtype=np.uint32)

        assert len(stats["n_masked"]) == len(stats[f"mean"])

        if len(suffix) > 0:
            stats = {f"{k}{suffix}" : v for k,v in stats.items()}

        if "exptime" not in stats:
            #only need to add exptime column if it isn't there already (i.e. bypass suffix)
            stats["exptime"] =  np.arange(im.shape[output.parallel_axint])
            assert len(stats["exptime"]) == len(stats[f"mean{suffix}"])

        return stats
        
