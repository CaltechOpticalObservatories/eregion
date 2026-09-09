import argparse
import os
from eregion.pipeline.engine import PipelineEngine

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline-config", required=True, help="Path to pipeline config")
parser.add_argument("--detector-config", required=True, help="Path to detector config")
parser.add_argument("--input-dir", required=True, help="Path to input directory")
parser.add_argument("--output-base-dir", required=True, help="Path to output base directory")

args = parser.parse_args()
outdir = os.path.join(args.output_base_dir, os.path.expanduser(args.input_dir).split('DTU_dettest/')[1])

eng = PipelineEngine(pipeline_config_input=args.pipeline_config,
                     runtime_variables={
                         'DETECTOR_CONFIG': args.detector_config,
                         'BIAS_INPUT_SOURCE': args.input_dir+'/*bias*',
                         'LINBIN_INPUT_SOURCE': args.input_dir+'/*linbin*',
                         'FLAT_INPUT_SOURCE': args.input_dir+'/*flat*',
                     })
eng.run()

res = eng.results['digital_flow.linbin_stats']
res.save(outdir)