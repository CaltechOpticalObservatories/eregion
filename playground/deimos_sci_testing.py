import argparse
import os
import glob2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eregion.tasks import ImageCreator
from eregion.tasks.calibration import CalibrationResult, MasterBias
from eregion.tasks.custom import guess_image_type_from_filename_DEIMOS, load_image_fits_DEIMOS
from eregion.tasks.preprocessing import BiasSubtraction, ScanSubtraction, SigmaClipMasking
import eregion.tasks.ptc as ptc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--sigma_lower", type=float, default=5)
    parser.add_argument("--sigma_upper", type=float, default=5)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--skip-correlations", action="store_true", default=False)
    parser.add_argument("--break_after", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    input_files = glob2.glob(os.path.join(args.input_dir,'*.fits'))[0]
    rawpath = os.path.dirname(input_files)
    outpath = rawpath.replace("/DTU_dettest/", "/DTU_detreduce/")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # check if there is master bias already in outpath
    if len(glob2.glob(os.path.join(outpath, 'master_bias/*'))) != 0:
        mb_res = CalibrationResult.load(outpath)
    else:
        bias_creator = ImageCreator(detector_config=args.detector_config)
        bias_res = bias_creator.run(input_source=os.path.join(rawpath, "*bias*.fits"),
                                    identifier_func=guess_image_type_from_filename_DEIMOS,
                                    fileloader_func=load_image_fits_DEIMOS,
                                    data_on_demand=True)
        mb_task = MasterBias(method='median')
        mb_res = mb_task.run(images=bias_res.data('type == "bias"'))
        mb_res.save(outpath)
        del bias_res

    # init tasks for ptc
    creator = ImageCreator(detector_config=args.detector_config, max_batch_size=args.max_batch_size)
    bias_sub = BiasSubtraction()
    oscan_sub = ScanSubtraction(which_scan="serial_overscan", method="median_by_axis")
    cr_mask = SigmaClipMasking(sigma_clip_args={"sigma_lower": args.sigma_lower, "sigma_upper": args.sigma_upper})
    psd_size = None if args.skip_correlations else 9
    ptc_task = ptc.PTC(psd_size=psd_size)

    count = 0
    for flpair in creator.lazy_run(
        input_source=os.path.join(rawpath, "*flat*.fits"),
        identifier_func=guess_image_type_from_filename_DEIMOS,
        fileloader_func=load_image_fits_DEIMOS,
        data_on_demand=True,
    ):
        flpair = bias_sub.run(images=flpair.data, master_bias=mb_res.master_bias)
        flpair = oscan_sub.run(images=flpair.data)
        flpair = cr_mask.run(images=flpair.data)
        ptcres = ptc_task.run(images=flpair.data)

        if count == 0:
            preproc_res = flpair
            ptc_res = ptcres
        else:
            preproc_res = preproc_res.combine(flpair)
            ptc_res = ptc_res.combine(ptcres)
        count += 1
        print("Pairs processed: ", count)
        if count >= args.break_after and args.break_after > 0:
            break

    ptc_res.save(outpath)


if __name__ == "__main__":
    main()
