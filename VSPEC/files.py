from pathlib import Path

N_ZFILL = 5


def check_and_build_dir(path):
    """check and build directory
    Check if a path exists, if not, create it

    Args:
        path (pathib.Path): path to check

    Returns:
        None
    """
    if path.exists():
        pass
    else:
        path.mkdir()

def build_directories(name_of_run,path=Path('.')):
    """build directories
    Build the file system for a run if VSPEC.
    By default it is run in `./<name_of_run>`

    Args:
        name_of_run (str): descriptive filename where all of the
            data of this run of VSPEC will be kept
        path=pathlib.Path('.') (pathlib.Path): base directory
    
    Returns:
        (dict): mapping to the paths of all directories relevent
            to this model run
    """
    parent_folder = path / name_of_run
    check_and_build_dir(parent_folder)
    data_folder = parent_folder / 'Data'
    check_and_build_dir(data_folder)
    binned_spectra_folder = data_folder / 'binned_data'
    check_and_build_dir(binned_spectra_folder)
    all_spectra_values_folder = data_folder / 'AllModelSpectraValues'
    check_and_build_dir(all_spectra_values_folder)
    PSG_combined_spectra_folder = data_folder / 'PSGCombinedSpectra'
    check_and_build_dir(PSG_combined_spectra_folder)
    PSG_thermal_spectra_folder = data_folder / 'PSGThermalSpectra'
    check_and_build_dir(PSG_thermal_spectra_folder)
    PSG_noise_folder = data_folder / 'PSGNoise'
    check_and_build_dir(PSG_noise_folder)
    PSG_layers_folder = data_folder / 'PSGLayers'
    check_and_build_dir(PSG_layers_folder)
    PSG_configs_folder = data_folder / 'PSGConfig'
    check_and_build_dir(PSG_configs_folder)

    directories_dict = {'parent':parent_folder,
                        'data':data_folder,
                        'binned':binned_spectra_folder,
                        'all_model': all_spectra_values_folder,
                        'psg_combined': PSG_combined_spectra_folder,
                        'psg_thermal': PSG_thermal_spectra_folder,
                        'psg_noise': PSG_noise_folder,
                        'psg_layers': PSG_layers_folder,
                        'psg_configs': PSG_configs_folder}
    return directories_dict
