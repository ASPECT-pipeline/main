from config import input_directory, output_directory, OBSERVPH, differential, pipeline, instrument, models, initGuess
import levels_012.main_calibration as main_calibration
import levels_012.modules.utilities as level_012_utilities
import level_3.main_level_3 as main_level_3
from pathlib import Path



"""
This is the main file to call the ASPECT data processing pipeline. More information about the pipeline in README.md file.

to run the pipeline: python3 ASPECT_calibration_pipeline/main_pipeline.py
"""


def main_pipeline():
    """
    The main function calls all calibration levels of the pipeline.

    """

    input_dir = Path(input_directory)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {input_dir}")
    
    output_dir = Path(output_directory)
    pipeline_steps = [str(s) for s in pipeline.split('-')]
    level_012_utilities.validate_pipeline_steps(pipeline_steps)

    if '1' in pipeline_steps:
        output_dir = Path(output_dir) / OBSERVPH # output directory for this acquisition
        output_dir.mkdir(parents=True, exist_ok=True) # create the directory for this acquisition
        print()
        print(f'New directory created for this acquisition: {output_dir.resolve()}')
        print()
        print(f'Executing pipeline levels 0 and 1')
        level_2_input = main_calibration.pipeline_levels_01(input_dir=input_dir, output_dir=output_dir, differential=differential)
        print(f'Calibration levels 0 and 1 completed. New Files created in directory: {level_2_input.resolve()}')
        file_list = [f for f in level_2_input.iterdir() if f.is_file()]
        for file in file_list:
            print(file.name)
    else: 
        level_2_input = input_dir
    
    instru = instrument.lower()
    
    if '2' in pipeline_steps:
        print(f'Executing pipeline level 2')
        level_2_output = main_calibration.pipeline_level_02(input_dir=level_2_input, output_dir=output_dir, instrument=instru)
        level_3_input = level_2_output
    else: 
        level_3_input = next(level_2_input.glob("*_2B.fits"), None)
    
    if '3' in pipeline_steps:
        print(f'Executing pipeline level 3')
        if level_3_input is not None and Path(level_3_input).is_file():
            level_3_output = main_level_3.level3(fits_file=level_3_input,output_dir=output_dir, instrument=instrument, models=models, initGuess=initGuess)
        else:
            raise FileNotFoundError(f"Level 3 expects a FITS file with '_2B.fits' ending, but got: {level_3_input}")


main_pipeline()