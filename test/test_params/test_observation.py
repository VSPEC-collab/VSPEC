import pytest
from astropy import units as u
import numpy as np

from VSPEC.config import flux_unit as default_flux_unit
from VSPEC.params.observation import BandpassParameters, ObservationParameters,ccdParameters,DetectorParameters,InstrumentParameters
from VSPEC.params.observation import TelescopeParameters, SingleDishParameters, CoronagraphParameters

def test_ObservationParameters_init():
    # Create ObservationParameters with valid values
    observation_time = 10 * u.hour
    integration_time = 1 * u.minute
    obs_params = ObservationParameters(observation_time, integration_time)

    # Perform assertions on the instance attributes
    assert obs_params.observation_time == observation_time
    assert obs_params.integration_time == integration_time

    # Create ObservationParameters with invalid values (integration time longer than observation time)
    invalid_observation_time = 1 * u.hour
    invalid_integration_time = 2 * u.hour

    # Check that ValueError is raised when creating ObservationParameters with invalid values
    with pytest.raises(ValueError):
        ObservationParameters(invalid_observation_time, invalid_integration_time)


def test_ObservationParameters_total_images():
    # Create ObservationParameters with observation time of 10 hours and integration time of 1 hour
    observation_time = 10 * u.hour
    integration_time = 1 * u.hour
    zodi = 1.0
    obs_params = ObservationParameters(observation_time, integration_time)

    # Calculate the expected total number of images
    expected_total_images = 10

    # Check that the calculated total_images property matches the expected value
    assert obs_params.total_images == expected_total_images

def test_ObservationParameters_from_dict():
    # Create a dictionary with the ObservationParameters data
    obs_params_data = {
        "observation_time": '10 day',
        "integration_time": '1 day',
    }

    # Create an ObservationParameters instance using the from_dict class method
    obs_params = ObservationParameters.from_dict(obs_params_data)

    # Perform assertions on the instance attributes
    assert obs_params.observation_time == 10 * u.day
    assert obs_params.integration_time == 1 * u.day





def test_BandpassParameters_from_dict():
    # Create a dictionary with the BandpassParameters data
    bandpass_params_data = {
        "wl_blue": '1.0 um',
        "wl_red": '18.0 um',
        "resolving_power": '50',
        "wavelength_unit": "um",
        "flux_unit": "W m-2 um-1"
    }

    # Create a BandpassParameters instance using the _from_dict class method
    bandpass_params = BandpassParameters.from_dict(bandpass_params_data)

    # Perform assertions on the instance attributes
    assert bandpass_params.wl_blue == 1.0 * u.um
    assert bandpass_params.wl_red == 18.0 * u.um
    assert bandpass_params.resolving_power == 50
    assert bandpass_params.wavelength_unit == u.um
    assert bandpass_params.flux_unit == u.Unit("W m-2 um-1")

def test_BandpassParameters_to_psg():
    # Create a BandpassParameters instance
    bandpass_params = BandpassParameters(
        wl_blue=1.0 * u.um,
        wl_red=18.0 * u.um,
        resolving_power=50,
        wavelength_unit=u.um,
        flux_unit=u.Unit("W m-2 um-1")
    )

    # Convert the BandpassParameters to the PSG input format
    psg_input = bandpass_params.to_psg()

    # Perform assertions on the PSG input dictionary
    assert psg_input["GENERATOR-RANGE1"] == "1.00"
    assert psg_input["GENERATOR-RANGE2"] == "18.00"
    assert psg_input["GENERATOR-RANGEUNIT"] == "um"
    assert psg_input["GENERATOR-RESOLUTION"] == "50"
    assert psg_input["GENERATOR-RESOLUTIONUNIT"] == "RP"
    assert psg_input["GENERATOR-RADUNITS"] == "Wm2um"

def test_BandpassParameters_mirecle():
    # Create a BandpassParameters instance using the mirecle class method
    bandpass_params = BandpassParameters.mirecle()

    # Perform assertions on the instance attributes
    assert bandpass_params.wl_blue == 1.0 * u.um
    assert bandpass_params.wl_red == 18.0 * u.um
    assert bandpass_params.resolving_power == 50
    assert bandpass_params.wavelength_unit == u.um
    assert bandpass_params.flux_unit == default_flux_unit


def test_ccdParameters_init():
    # Create ccdParameters with valid values
    pixel_sampling = 64
    read_noise = 6 * u.electron
    dark_current = 100 * u.electron / u.s
    throughput = 0.5
    emissivity = 0.1
    temperature = 35 * u.K
    ccd_params = ccdParameters(pixel_sampling, read_noise, dark_current, throughput, emissivity, temperature)

    # Perform assertions on the instance attributes
    assert ccd_params.pixel_sampling == pixel_sampling
    assert ccd_params.read_noise == read_noise
    assert ccd_params.dark_current == dark_current
    assert ccd_params.throughput == throughput
    assert ccd_params.emissivity == emissivity
    assert ccd_params.temperature == temperature

def test_ccdParameters_to_psg():
    # Create ccdParameters with specific values
    pixel_sampling = 64
    read_noise = 6 * u.electron
    dark_current = 100 * u.electron / u.s
    throughput = 0.5
    emissivity = 0.1
    temperature = 35 * u.K
    ccd_params = ccdParameters(pixel_sampling, read_noise, dark_current, throughput, emissivity, temperature)

    # Define the expected PSG input format dictionary
    expected_dict = {
        'GENERATOR-NOISE': 'CCD',
        'GENERATOR-NOISEPIXELS': f'{pixel_sampling}',
        'GENERATOR-NOISE1': f'{read_noise.to_value(u.electron):.1f}',
        'GENERATOR-NOISE2': f'{dark_current.to_value(u.electron/u.s):.1f}',
        'GENERATOR-NOISEOEFF': f'{throughput:.2f}',
        'GENERATOR-NOISEOEMIS': f'{emissivity:.2f}',
        'GENERATOR-NOISEOTEMP': f'{temperature.to_value(u.K):.1f}'
    }

    # Check that the to_psg method returns the expected dictionary
    assert ccd_params.to_psg() == expected_dict

def test_ccdParameters_from_dict():
    # Create a dictionary with the ccdParameters data
    ccd_params_data = {
        "pixel_sampling": '64',
        "read_noise": '6 electron',
        "dark_current": '100 electron / s',
        "throughput": '0.5',
        "emissivity": '0.1',
        "temperature": '35 K'
    }

    # Create a ccdParameters instance using the _from_dict class method
    ccd_params = ccdParameters.from_dict(ccd_params_data)

    # Perform assertions on the instance attributes
    assert ccd_params.pixel_sampling == 64
    assert ccd_params.read_noise == 6 * u.electron
    assert ccd_params.dark_current == 100 * u.electron / u.s
    assert ccd_params.throughput == 0.5
    assert ccd_params.emissivity == 0.1
    assert ccd_params.temperature == 35 * u.K

def test_ccdParameters_mirecle():
    # Create a ccdParameters instance using the mirecle class method
    ccd_params = ccdParameters.mirecle()

    # Perform assertions on the instance attributes based on the MIRECLE setup
    assert ccd_params.pixel_sampling == 64
    assert ccd_params.read_noise == 6 * u.electron
    assert ccd_params.dark_current == 100 * u.electron / u.s
    assert ccd_params.throughput == 0.5
    assert ccd_params.emissivity == 0.1
    assert ccd_params.temperature == 35 * u.K


def test_DetectorParameters_init():
    # Create CCD parameters for the detector
    pixel_sampling = 64
    read_noise = 6 * u.electron
    dark_current = 100 * u.electron / u.s
    throughput = 0.5
    emissivity = 0.1
    temperature = 35 * u.K
    ccd_params = ccdParameters(pixel_sampling, read_noise, dark_current, throughput, emissivity, temperature)

    # Create DetectorParameters with valid values
    beam_width = 5 * u.arcsec
    integration_time = 0.5 * u.s
    detector_params = DetectorParameters(beam_width, integration_time, ccd_params)

    # Perform assertions on the instance attributes
    assert detector_params.beam_width == beam_width
    assert detector_params.integration_time == integration_time
    assert detector_params.ccd == ccd_params

def test_DetectorParameters_to_psg():
    # Create CCD parameters for the detector
    pixel_sampling = 64
    read_noise = 6 * u.electron
    dark_current = 100 * u.electron / u.s
    throughput = 0.5
    emissivity = 0.1
    temperature = 35 * u.K
    ccd_params = ccdParameters(pixel_sampling, read_noise, dark_current, throughput, emissivity, temperature)

    # Create DetectorParameters with specific values
    beam_width = 5 * u.arcsec
    integration_time = 0.5 * u.s
    detector_params = DetectorParameters(beam_width, integration_time, ccd_params)

    # Define the expected PSG input format dictionary
    expected_dict = {
        'GENERATOR-BEAM': f'{beam_width.to_value(u.arcsec):.4f}',
        'GENERATOR-BEAM-UNIT': 'arcsec',
        'GENERATOR-NOISE': 'CCD',
        'GENERATOR-NOISEPIXELS': f'{pixel_sampling}',
        'GENERATOR-NOISE1': f'{read_noise.to_value(u.electron):.1f}',
        'GENERATOR-NOISE2': f'{dark_current.to_value(u.electron/u.s):.1f}',
        'GENERATOR-NOISEOEFF': f'{throughput:.2f}',
        'GENERATOR-NOISEOEMIS': f'{emissivity:.2f}',
        'GENERATOR-NOISEOTEMP': f'{temperature.to_value(u.K):.1f}'
    }

    # Check that the to_psg method returns the expected dictionary
    assert detector_params.to_psg() == expected_dict

def test_DetectorParameters_from_dict():
    # Create a dictionary with the detector parameters data
    detector_params_data = {
        "beam_width": '5 arcsec',
        "integration_time": '0.5 s',
        "ccd": {
            "pixel_sampling": '64',
            "read_noise": '6 electron',
            "dark_current": '100 electron / s',
            "throughput": '0.5',
            "emissivity": '0.1',
            "temperature": '35 K'
        }
    }

    # Create DetectorParameters instance using the _from_dict class method
    detector_params = DetectorParameters.from_dict(detector_params_data)

    # Perform assertions on the instance attributes
    assert detector_params.beam_width == 5 * u.arcsec
    assert detector_params.integration_time == 0.5 * u.s
    assert detector_params.ccd.pixel_sampling == 64
    assert detector_params.ccd.read_noise == 6 * u.electron
    assert detector_params.ccd.dark_current == 100 * u.electron / u.s
    assert detector_params.ccd.throughput == 0.5
    assert detector_params.ccd.emissivity == 0.1
    assert detector_params.ccd.temperature == 35 * u.K

def test_DetectorParameters_mirecle():
    # Create DetectorParameters instance using the mirecle class method
    detector_params = DetectorParameters.mirecle()

    # Perform assertions on the instance attributes
    assert detector_params.beam_width == 5 * u.arcsec
    assert detector_params.integration_time == 0.5 * u.s
    assert detector_params.ccd.pixel_sampling == 64
    assert detector_params.ccd.read_noise == 6 * u.electron
    assert detector_params.ccd.dark_current == 100 * u.electron / u.s
    assert detector_params.ccd.throughput == 0.5
    assert detector_params.ccd.emissivity == 0.1
    assert detector_params.ccd.temperature == 35 * u.K


@pytest.fixture
def single_dish_parameters():
    return SingleDishParameters(aperture=4 * u.m, zodi=10.0)

@pytest.fixture
def coronagraph_parameters():
    iwa_x = np.array([1.0, 2.0, 3.0])
    iwa_y = np.array([0.1, 0.2, 0.3])
    return CoronagraphParameters(
        aperture=8 * u.m,
        zodi=5.0,
        exozodi=2.0,
        contrast=1e-6,
        iwa_x=iwa_x,
        iwa_y=iwa_y,
    )

def test_single_dish_parameters(single_dish_parameters):
    assert single_dish_parameters.aperture == 4 * u.m
    assert single_dish_parameters.mode == 'single'
    assert single_dish_parameters.zodi == 10.0

    psg_config = single_dish_parameters.to_psg()
    assert psg_config['GEOMETRY'] == 'Observatory'
    assert psg_config['GENERATOR-DIAMTELE'] == '4.00'
    assert psg_config['GENERATOR-TELESCOPE'] == 'SINGLE'
    assert psg_config['GENERATOR-TELESCOPE2'] == '10.00'

def test_single_dish_parameters_mirecle():
    mirecle_parameters = SingleDishParameters.mirecle()
    assert mirecle_parameters.aperture == 2 * u.m
    assert mirecle_parameters.mode == 'single'
    assert mirecle_parameters.zodi == 1.0

def test_coronagraph_parameters(coronagraph_parameters:CoronagraphParameters):
    assert coronagraph_parameters.aperture == 8 * u.m
    assert coronagraph_parameters.mode == 'coronagraph'
    assert coronagraph_parameters.zodi == 5.0
    assert coronagraph_parameters.exozodi == 2.0
    assert coronagraph_parameters.contrast == 1e-6
    assert np.array_equal(coronagraph_parameters.iwa_x, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(coronagraph_parameters.iwa_y, np.array([0.1, 0.2, 0.3]))

    psg_config = coronagraph_parameters.to_psg()
    assert psg_config['GEOMETRY'] == 'Observatory'
    assert psg_config['GENERATOR-DIAMTELE'] == '8.00'
    assert psg_config['GENERATOR-TELESCOPE'] == 'CORONA'
    assert psg_config['GENERATOR-TELESCOPE2'] == '5.00,2.00'
    assert psg_config['GENERATOR-TELESCOPE1'] == '1.00e-06'
    assert psg_config['GENERATOR-TELESCOPE3'] == '1.00e+00@1.00e-01,2.00e+00@2.00e-01,3.00e+00@3.00e-01'

def test_from_dict_single_dish_parameters():
    d = {
        'aperture': '4.0 m',
        'zodi': '10.0',
    }
    parameters = SingleDishParameters.from_dict(d)
    assert parameters.aperture == 4 * u.m
    assert parameters.zodi == 10.0

def test_from_dict_coronagraph_parameters():
    d = {
        'aperture': '8.0 m',
        'zodi': '5.0',
        'exozodi': '2.0',
        'contrast': '1e-6',
        'iwa_x': '1.0,2.0,3.0',
        'iwa_y': '0.1,0.2,0.3',
    }
    parameters = CoronagraphParameters._from_dict(d)
    assert parameters.aperture == 8 * u.m
    assert parameters.zodi == 5.0
    assert parameters.exozodi == 2.0
    assert parameters.contrast == 1e-6
    assert np.array_equal(parameters.iwa_x, np.array([1.0, 2.0, 3.0],dtype='float32'))
    assert np.array_equal(parameters.iwa_y, np.array([0.1, 0.2, 0.3],dtype='float32'))



def test_InstrumentParameters_init():
    # Create bandpass parameters for the instrument
    wl_blue = 1 * u.um
    wl_red = 2 * u.um
    resolving_power = 100
    wavelength_unit = u.um
    flux_unit = u.Unit('W m-2 um-1')
    bandpass_params = BandpassParameters(wl_blue, wl_red, resolving_power, wavelength_unit, flux_unit)

    # Create CCD parameters for the detector
    beam_width = 5 * u.arcsec
    integration_time = 0.5 * u.s
    pixel_sampling = 64
    read_noise = 6 * u.electron
    dark_current = 100 * u.electron / u.s
    throughput = 0.5
    emissivity = 0.1
    temperature = 35 * u.K
    ccd_params = ccdParameters(pixel_sampling, read_noise, dark_current, throughput, emissivity, temperature)

    # Create DetectorParameters with valid values
    detector_params = DetectorParameters(beam_width, integration_time, ccd_params)

    # Create InstrumentParameters with valid values
    telescope = SingleDishParameters(2*u.m,1.5)
    instrument_params = InstrumentParameters(telescope, bandpass_params, detector_params)

    # Perform assertions on the instance attributes
    assert instrument_params.telescope.aperture == 2*u.m
    assert instrument_params.bandpass == bandpass_params
    assert instrument_params.detector == detector_params

def test_InstrumentParameters_to_psg():
    # Create bandpass parameters for the instrument
    wl_blue = 1 * u.um
    wl_red = 2 * u.um
    resolving_power = 100
    wavelength_unit = u.um
    flux_unit = u.Unit('W m-2 um-1')
    bandpass_params = BandpassParameters(wl_blue, wl_red, resolving_power, wavelength_unit, flux_unit)

    # Create CCD parameters for the detector
    beam_width = 5 * u.arcsec
    integration_time = 0.5 * u.s
    pixel_sampling = 64
    read_noise = 6 * u.electron
    dark_current = 100 * u.electron / u.s
    throughput = 0.5
    emissivity = 0.1
    temperature = 35 * u.K
    ccd_params = ccdParameters(pixel_sampling, read_noise, dark_current, throughput, emissivity, temperature)

    # Create DetectorParameters with specific values
    detector_params = DetectorParameters(beam_width, integration_time, ccd_params)

    # Create InstrumentParameters with specific values
    telescope = SingleDishParameters(2*u.m,1.5)
    instrument_params = InstrumentParameters(telescope, bandpass_params, detector_params)

    # Define the expected PSG input format dictionary
    expected_dict = {
        'GEOMETRY': 'Observatory',
        'GENERATOR-DIAMTELE': f'{(2*u.m).to_value(u.m):.2f}',
        'GENERATOR-TELESCOPE': 'SINGLE',
        'GENERATOR-TELESCOPE2' : '1.50',
        'GENERATOR-RANGE1': f'{wl_blue.to_value(wavelength_unit):.2f}',
        'GENERATOR-RANGE2': f'{wl_red.to_value(wavelength_unit):.2f}',
        'GENERATOR-RANGEUNIT': wavelength_unit.to_string(),
        'GENERATOR-RESOLUTION': f'{resolving_power}',
        'GENERATOR-RESOLUTIONUNIT': 'RP',
        'GENERATOR-RADUNITS': 'Wm2um',
        'GENERATOR-BEAM': f'{beam_width.to_value(u.arcsec):.4f}',
        'GENERATOR-BEAM-UNIT': 'arcsec',
        'GENERATOR-NOISE': 'CCD',
        'GENERATOR-NOISEPIXELS': f'{pixel_sampling}',
        'GENERATOR-NOISE1': f'{read_noise.to_value(u.electron)}',
        'GENERATOR-NOISE2': f'{dark_current.to_value(u.electron/u.s)}',
        'GENERATOR-NOISEOEFF': f'{throughput:.2f}',
        'GENERATOR-NOISEOTEMP': f'{temperature.to_value(u.K):.1f}',
        'GENERATOR-NOISEOEMIS': f'{emissivity:.2f}',
    }

    # Check if the to_psg method produces the expected dictionary
    assert instrument_params.to_psg() == expected_dict

def test_InstrumentParameters_mirecle():
    # Call the mirecle class method
    instrument_params = InstrumentParameters.mirecle()

    # Perform assertions on the instance attributes
    assert instrument_params.telescope.aperture == 2 * u.m

    # Check if the bandpass and detector parameters are initialized correctly
    assert isinstance(instrument_params.bandpass, BandpassParameters)
    assert instrument_params.bandpass.wl_blue == 1 * u.um
    assert instrument_params.bandpass.wl_red == 18 * u.um
    assert instrument_params.bandpass.resolving_power == 50
    assert instrument_params.bandpass.wavelength_unit == u.um

    assert isinstance(instrument_params.detector, DetectorParameters)
    assert instrument_params.detector.beam_width == 5 * u.arcsec
    assert instrument_params.detector.integration_time == 0.5 * u.s
    assert instrument_params.detector.ccd.pixel_sampling == 64
    assert instrument_params.detector.ccd.read_noise == 6 * u.electron
    assert instrument_params.detector.ccd.dark_current == 100 * u.electron / u.s
    assert instrument_params.detector.ccd.throughput == 0.5
    assert instrument_params.detector.ccd.emissivity == 0.1
    assert instrument_params.detector.ccd.temperature == 35 * u.K


def test_InstrumentParameters_from_dict():
    # Create a dictionary representing instrument parameters
    instrument_dict = {
        'single': {
            'aperture': '2.5 m',
            'zodi': '1.0'
        },
        'bandpass': {
            'wl_blue': '1.2 um',
            'wl_red': '2.4 um',
            'resolving_power': '80',
            'wavelength_unit': 'um',
            'flux_unit': 'W m-2 um-1'
        },
        'detector': {
            'beam_width': '6.5 arcsec',
            'integration_time': '0.8 s',
            'ccd': {
                'pixel_sampling': '128',
                'read_noise': '8.5 electron',
                'dark_current': '120 electron / s',
                'throughput': '0.7',
                'emissivity': '0.2',
                'temperature': '30 K'
            }
        }
    }

    # Create InstrumentParameters instance from the dictionary
    instrument_params = InstrumentParameters.from_dict(instrument_dict)

    # Perform assertions on the instance attributes
    assert instrument_params.telescope.aperture == 2.5 * u.m

    # Check if the bandpass parameters are initialized correctly
    assert isinstance(instrument_params.bandpass, BandpassParameters)
    assert instrument_params.bandpass.wl_blue == 1.2 * u.um
    assert instrument_params.bandpass.wl_red == 2.4 * u.um
    assert instrument_params.bandpass.resolving_power == 80
    assert instrument_params.bandpass.wavelength_unit == u.um
    assert instrument_params.bandpass.flux_unit == u.Unit('W m-2 um-1')

    # Check if the detector parameters are initialized correctly
    assert isinstance(instrument_params.detector, DetectorParameters)
    assert instrument_params.detector.beam_width == 6.5 * u.arcsec
    assert instrument_params.detector.integration_time == 0.8 * u.s
    assert instrument_params.detector.ccd.pixel_sampling == 128
    assert instrument_params.detector.ccd.read_noise == 8.5 * u.electron
    assert instrument_params.detector.ccd.dark_current == 120 * u.electron / u.s
    assert instrument_params.detector.ccd.throughput == 0.7
    assert instrument_params.detector.ccd.emissivity == 0.2
    assert instrument_params.detector.ccd.temperature == 30 * u.K

