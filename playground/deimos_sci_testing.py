import argparse
import os
import glob2
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from datamodels import ImageBundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eregion.tasks import ImageCreator
from eregion.tasks.calibration import CalibrationResult, MasterBias
from eregion.tasks.custom import guess_image_type_from_filename_DEIMOS, load_image_fits_DEIMOS
from eregion.tasks.preprocessing import BiasSubtraction, ScanSubtraction, SigmaClipMasking
import eregion.tasks.ptc as ptc

def _plot_panels(ax, df, cols, x_col, yscale, xscale):
    axkey = {"det_1": ax[0,3], "det_2": ax[0,2], "det_3": ax[0,1], "det_4": ax[0,0],
             "det_5": ax[1,0], "det_6": ax[1,1], "det_7": ax[1,2], "det_8": ax[1,3]}
    markers_list = list(Line2D.markers.keys())

    for det_id, axs in axkey.items():
        axs.clear()
        for out, color in zip(['E', 'F'], ['blue', 'red']):
            dfsub = df[(df['det_id']==det_id) & (df['output']==out)]
            for i,col in enumerate(cols):
                axs.plot(list(dfsub[x_col]), list(dfsub[col]), color=color, label=col+'_'+out, marker=markers_list[i],
                         alpha=0.7)

        axs.set_xlabel(x_col, fontsize=10)
        axs.set_ylabel('PTC column', fontsize=10)
        axs.set_title(det_id, fontsize=12)
        axs.grid(True)
        try:
            axs.set_xscale(xscale)
        except:
            pass
        try:
            axs.set_yscale(yscale)
        except:
            pass
    ax[0, 3].legend(ncols=1, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0)
    return ax

def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots(2,4, figsize=(20,8), tight_layout=True)
    return fig, ax

def update_live_plot(fig, ax, ptc_table, cols, x_col='exptime',
                     yscale='linear', xscale='linear'):
    if not cols:
        return
    df = ptc_table.copy()
    ax = _plot_panels(ax, df, cols, x_col, yscale, xscale)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

def ptc_plot(runpath, cols, x_col='exptime', yscale='linear', xscale='linear'):
    ptc_res = ptc.PTCResult.load(runpath)
    df = ptc_res.ptc_table.copy()
    del ptc_res

    fig, ax = plt.subplots(2, 4, figsize=(20,8), tight_layout=True)
    ax = _plot_panels(ax, df, cols, x_col, yscale, xscale)
    plt.show()


def main():
    args = parse_args()

    fig = ax = None
    if args.live_plot:
        fig, ax = init_live_plot()

    input_files = glob2.glob(os.path.join(args.input_dir,'*.fits'))[0]
    rawpath = os.path.dirname(input_files)
    outpath = os.path.join(args.output_base_dir, rawpath.split("DTU_dettest/")[1])
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
        mb_res = mb_task.run(bias_images=bias_res.data('type == "bias"'))
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

        del flpair

        if count == 0:
            # preproc_res = flpair
            ptc_res = ptcres
        else:
            # preproc_res = preproc_res.combine(flpair)
            ptc_res = ptc_res.combine(ptcres)
        count += 1
        print("Pairs processed: ", count)

        if count % 20 == 0:
            ptc_res.save(outpath)
            del ptc_res.diff_images
            ptc_res.diff_images = ImageBundle()
            if args.live_plot:
                update_live_plot(fig, ax, ptc_res.ptc_table, args.plot_cols, args.plot_x, args.yscale, args.xscale)

        if count >= args.break_after and args.break_after > 0:
            break

    ptc_res.save(outpath)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Run PTC analysis on DEIMOS data")
    parser.add_argument("--detector-config", required=True, help="Path to detector config")
    parser.add_argument("--input-dir", required=True, help="Path to input directory")
    parser.add_argument("--output-base-dir", required=True, help="Path to output base directory")
    parser.add_argument("--sigma_lower", type=float, default=5, help="Lower sigma for clipping")
    parser.add_argument("--sigma_upper", type=float, default=5, help="Upper sigma for clipping")
    parser.add_argument("--max-batch-size", type=int, default=2, help="Maximum batch size for lazy run")
    parser.add_argument("--skip-correlations", action="store_true", default=False, help="Skip correlation calculations")
    parser.add_argument("--break_after", type=int, default=0, help="Break after processing this many pairs, set to 0 to disable")
    parser.add_argument("--live_plot", action="store_true", default=False, help="Do live plotting")
    parser.add_argument("--plot_cols", nargs="+", default=["variance"], help="Columns to plot")
    parser.add_argument("--plot-x", default="mean", help="Column to plot on x-axis")
    parser.add_argument("--yscale", type=str, default='log', help="Scale for y-axis")
    parser.add_argument("--xscale", type=str, default='log', help="Scale for x-axis")
    return parser.parse_args()

if __name__ == "__main__":
    main()
