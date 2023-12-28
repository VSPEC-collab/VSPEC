"""
Observation parameters
"""
from typing import Union
from astropy import units as u
import yaml

from VSPEC.config import flux_unit as default_flux_unit, PRESET_PATH
from VSPEC.params.base import BaseParameters, PSGtable, parse_table


class ObservationParameters(BaseParameters):
    """
    Class storing parameters for observations.

    Parameters
    ----------
    observation_time : astropy.units.Quantity
        The total duration of the observation.
    integration_time : astropy.units.Quantity
        The integration time of each epoch of observation.
    

    Attributes
    ----------
    total_images
    observation_time : astropy.units.Quantity
        The total duration of the observation.
    integration_time : astropy.units.Quantity
        The integration time of each epoch of observation.

    Raises
    ------
    ValueError
        If the integration time is longer than the total observation time.

    """

    def __init__(
        self,
        observation_time: u.Quantity,
        integration_time: u.Quantity,
    ):
        self.observation_time = observation_time
        self.integration_time = integration_time
        self._validate()

    def _validate(self):
        if self.integration_time > self.observation_time:
            raise ValueError(
                'Length of integrations cannot be longer than the total observation.')

    @property
    def total_images(self) -> int:
        """
        Total number of images based on the observation time and integration time.

        Returns
        -------
        int
            The total number of images.

        """

        return int(round((self.observation_time/self.integration_time).to_value(u.dimensionless_unscaled)))
    
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            observation_time = u.Quantity(d['observation_time']),
            integration_time = u.Quantity(d['integration_time']),
        )

class BandpassParameters(BaseParameters):
    """
    Class to store bandpass parameters for observations.

    Parameters
    ----------
    wl_blue : astropy.units.Quantity
        The shorter wavelength limit of the bandpass.
    wl_red : astropy.units.Quantity
        The longer wavelength limit of the bandpass.
    resolving_power : int
        The resolving power of the observation.
    wavelength_unit : astropy.units.Unit
        The unit of wavelength used for the bandpass parameters.
    flux_unit : astropy.units.Unit
        The unit of flux used for the bandpass parameters.

    Attributes
    ----------
    wl_blue : astropy.units.Quantity
        The shorter wavelength limit of the bandpass.
    wl_red : astropy.units.Quantity
        The longer wavelength limit of the bandpass.
    resolving_power : int
        The resolving power of the observation.
    wavelength_unit : astropy.units.Unit
        The unit of wavelength used for the bandpass parameters.
    flux_unit : astropy.units.Unit
        The unit of flux used for the bandpass parameters.
    """

    psg_rad_mapper = {u.Unit('W m-2 um-1'): 'Wm2um'}

    def __init__(
        self,
        wl_blue: u.Quantity,
        wl_red: u.Quantity,
        resolving_power: int,
        wavelength_unit: u.Unit,
        flux_unit: u.Unit
    ):
        self.wl_blue = wl_blue
        self.wl_red = wl_red
        self.resolving_power = resolving_power
        self.wavelength_unit = wavelength_unit
        self.flux_unit = flux_unit
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            wl_blue = u.Quantity(d['wl_blue']),
            wl_red = u.Quantity(d['wl_red']),
            resolving_power = int(d['resolving_power']),
            wavelength_unit = u.Unit(d['wavelength_unit']),
            flux_unit = u.Unit(d['flux_unit'])
        )
    def to_psg(self):
        """
        Convert the bandpass parameters to the PSG input format.

        Returns
        -------
        dict
            A dictionary representing the bandpass parameters in the PSG input format.

        """
        return {
            'GENERATOR-RANGE1': f'{self.wl_blue.to_value(self.wavelength_unit):.2f}',
            'GENERATOR-RANGE2': f'{self.wl_red.to_value(self.wavelength_unit):.2f}',
            'GENERATOR-RANGEUNIT': self.wavelength_unit.to_string(),
            'GENERATOR-RESOLUTION': f'{self.resolving_power}',
            'GENERATOR-RESOLUTIONUNIT': 'RP',
            'GENERATOR-RADUNITS': self.psg_rad_mapper[self.flux_unit]
        }
    @classmethod
    def mirecle(cls):
        """
        MIRECLE setup [1].
        Returns a `BandpassParameters` instance initialized with the MIRECLE setup.

        Returns
        -------
        BandpassParameters:
            `BandpassParameters` instance with the MIRECLE setup.

        References
        ----------
        [1] :cite:t:`2022AJ....164..176M`
        """
        return cls(
            1*u.um,
            18*u.um,
            50,
            u.um,
            default_flux_unit
        )
    @classmethod
    def miri_lrs(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['bandpass']['miri-lrs'])
    @classmethod
    def niriss_soss(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['bandpass']['niriss-soss'])


class ccdParameters(BaseParameters):
    """
    Class to store CCD parameters for observations.

    Parameters
    ----------
    pixel_sampling : int
        The pixel sampling of the CCD.
    read_noise : astropy.units.Quantity
        The read noise of the CCD in electrons.
    dark_current : astropy.units.Quantity
        The dark current of the CCD in electrons/second.
    throughput : float
        The throughput of the CCD.
    emissivity : float
        The emissivity of the CCD.
    temperature : astropy.units.Quantity
        The temperature of the CCD.


    Attributes
    ----------
    pixel_sampling : int
        The pixel sampling of the CCD.
    read_noise : astropy.units.Quantity
        The read noise of the CCD in electrons.
    dark_current : astropy.units.Quantity
        The dark current of the CCD in electrons/second.
    throughput : float
        The throughput of the CCD.
    emissivity : float
        The emissivity of the CCD.
    temperature : astropy.units.Quantity
        The temperature of the CCD.
    """

    def __init__(
        self,
        pixel_sampling: int,
        read_noise: u.Quantity,
        dark_current: u.Quantity,
        throughput: float,
        emissivity: float,
        temperature: u.Quantity
    ):
        self.pixel_sampling = pixel_sampling
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.throughput = throughput
        self.emissivity = emissivity
        self.temperature = temperature
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            pixel_sampling = parse_table(d['pixel_sampling'],int),
            read_noise = parse_table(d['read_noise'],u.Quantity),
            dark_current = parse_table(d['dark_current'],u.Quantity),
            throughput = parse_table(d['throughput'],float),
            emissivity = float(d['emissivity']),
            temperature = u.Quantity(d['temperature'])
        )
    def to_psg(self)->dict:
        """
        Convert the CCD parameters to the PSG input format.

        Returns
        -------
        dict
            A dictionary representing the CCD parameters in the PSG input format.

        """
        return {
            'GENERATOR-NOISE': 'CCD',
            'GENERATOR-NOISEPIXELS': f'{self.pixel_sampling}',
            'GENERATOR-NOISE1': f'{self.read_noise.to_value(u.electron):.1f}' if isinstance(self.read_noise,u.Quantity) else str(self.read_noise),
            'GENERATOR-NOISE2': f'{self.dark_current.to_value(u.electron/u.s):.1f}' if isinstance(self.dark_current,u.Quantity) else str(self.dark_current),
            'GENERATOR-NOISEOEFF': f'{self.throughput:.2f}' if isinstance(self.throughput,float) else str(self.throughput),
            'GENERATOR-NOISEOEMIS': f'{self.emissivity:.2f}',
            'GENERATOR-NOISEOTEMP': f'{self.temperature.to_value(u.K):.1f}'
        }

    @classmethod
    def mirecle(cls):
        """
        MIRECLE setup [1].
        Returns a CCDParameters instance initialized with the MIRECLE setup.

        Returns
        -------
        CCDParameters:
            CCDParameters instance for the MIRECLE setup.

        References
        ----------
        [1] :cite:t:`2022AJ....164..176M`
        """

        return cls(
            pixel_sampling=64,
            read_noise=6 * u.electron,
            dark_current=100 * u.electron / u.s,
            throughput=0.5,
            emissivity=0.1,
            temperature=35*u.K
        )
    @classmethod
    def miri_lrs(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['ccd']['miri-lrs'])
    @classmethod
    def niriss_soss(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['ccd']['niriss-soss'])



class DetectorParameters(BaseParameters):
    """
    Class to store detector parameters for observations.

    Parameters
    ----------
    beam_width : astropy.units.Quantity
        The beam width of the detector.
    integration_time : astropy.units.Quantity
        The integration time of the detector.
    ccd : ccdParameters
        The CCD parameters for the detector.

    Attributes
    ----------
    beam_width : astropy.units.Quantity
        The beam width of the detector.
    integration_time : astropy.units.Quantity
        The integration time of the detector.
    ccd : ccdParameters
        The CCD parameters for the detector.
    """

    def __init__(
        self,
        beam_width: u.Quantity,
        integration_time: u.Quantity,
        ccd: ccdParameters
    ):
        self.beam_width = beam_width
        self.integration_time = integration_time
        self.ccd = ccd
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            beam_width = u.Quantity(d['beam_width']),
            integration_time = u.Quantity(d['integration_time']),
            ccd = ccdParameters.from_dict(d['ccd']),
        )
    def to_psg(self)->dict:
        """
        Convert the detector parameters to the PSG input format.

        Returns
        -------
        dict
            A dictionary representing the detector parameters in the PSG input format.

        """
        config = {
            'GENERATOR-BEAM': f'{self.beam_width.to_value(u.arcsec):.4f}',
            'GENERATOR-BEAM-UNIT': 'arcsec'
        }
        config.update(self.ccd.to_psg())
        return config
    @classmethod
    def miri_lrs(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['detector']['miri-lrs'])
    @classmethod
    def niriss_soss(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['detector']['niriss-soss'])

    @classmethod
    def mirecle(cls):
        """
        MIRECLE setup [1].
        Returns a DetectorParameters instance initialized with the MIRECLE setup.

        Returns
        -------
        DetectorParameters:
            DetectorParameters instance for the MIRECLE setup.

        References
        ----------
        [1] :cite:t:`2022AJ....164..176M`
        """
        return cls(
            beam_width=5*u.arcsec,
            integration_time=0.5*u.s,
            ccd=ccdParameters.mirecle()
        )

class TelescopeParameters(BaseParameters):
    """
    Base class for telescope Parameters.

    Parameters
    ----------
    aperture : astropy.units.Quantity
        The aperture size of the telescope.
    mode : str
        The mode of the telescope. Valid values are 'single' or 'coronagraph'.
    zodi : float
        The level of the zodiacal background. Acceptable values range from
        1.0 (Ecliptic pole/minimum) to 100.0 (Close to ecliptic/Sun).

    Other Parameters
    ----------------
    kwargs
        Additional parameters specific to the telescope.

    Attributes
    ----------
    aperture : astropy.units.Quantity
        The aperture size of the telescope.
    mode : str
        The mode of the telescope.
    zodi : float
        The level of the zodiacal background.
    """


    _mode_translator = {
        'single': 'SINGLE',
        'coronagraph': 'CORONA',
    }
    def __init__(
        self,
        aperture:u.Quantity,
        mode:str,
        zodi:float,
        **kwargs
    ):
        self.aperture = aperture
        self.mode = mode
        self.zodi = zodi
        for key,value in kwargs.items():
            self.__setattr__(key,value)
    def _to_psg(self):
        return {}
    def to_psg(self):
        """
        Convert telescope parameters to PSG format.

        Returns
        -------
        dict
            A dictionary containing the PSG configuration for the telescope.
        """

        config = {
            'GEOMETRY': 'Observatory',
            'GENERATOR-DIAMTELE': f'{self.aperture.to_value(u.m):.2f}',
            'GENERATOR-TELESCOPE': self._mode_translator[self.mode],
        }
        config.update(self._to_psg())
        return config

class SingleDishParameters(TelescopeParameters):
    """
    Parameters for a single dish telescope.

    Parameters
    ----------
    aperture : astropy.units.Quantity
        The aperture size of the telescope.
    zodi : float
        The level of the zodiacal background. Acceptable values range from
        1.0 (Ecliptic pole/minimum) to 100.0 (Close to ecliptic/Sun).
    """

    def __init__(self, aperture: u.Quantity,zodi:float):
        super().__init__(aperture, 'single',zodi)
    def _to_psg(self):
        return {
            'GENERATOR-TELESCOPE2': f'{self.zodi:.2f}'
        }
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            aperture = u.Quantity(d['aperture']),
            zodi = float(d['zodi'])
        )
    @classmethod
    def mirecle(cls):
        """
        Create a SingleDishParameters instance with MIRECLE parameters [1].

        Returns
        -------
        SingleDishParameters
            The created SingleDishParameters instance with MIRECLE parameters.
        
        References
        ----------
        [1] :cite:t:`2022AJ....164..176M`
        """

        return cls(2*u.m, 1.0)
    @classmethod
    def jwst(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['telescope']['jwst']['single'])

class CoronagraphParameters(TelescopeParameters):
    """
    Parameters for a coronagraph telescope.

    Parameters
    ----------
    aperture : astropy.units.Quantity
        The aperture size of the telescope.
    zodi : float
        The level of the zodiacal background. Acceptable values range from
        1.0 (Ecliptic pole/minimum) to 100.0 (Close to ecliptic/Sun).
    exozodi : float
        The level of the exozodiacal background.
    contrast : float
        The contrast level.
    IWA : PSGtable
        The inner working angle of the coronagraph
    """

    def __init__(
        self,
        aperture: u.Quantity,
        zodi: float,
        contrast: float,
        iwa: PSGtable
    ):
        super().__init__(
            aperture,
            'coronagraph',
            zodi,
            contrast = contrast,
            iwa = iwa
        )
    def _to_psg(self):
        return {
            'GENERATOR-TELESCOPE2': f'{self.zodi:.2f}',
            'GENERATOR-TELESCOPE1': f'{self.contrast:.2e}',
            'GENERATOR-TELESCOPE3': str(self.iwa)
        }
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            aperture = u.Quantity(d['aperture']),
            zodi = float(d['zodi']),
            contrast = float(d['contrast']),
            iwa = PSGtable.from_dict(d['iwa']['table'])
        )



class InstrumentParameters(BaseParameters):
    """
    Class to store instrument parameters for observations.

    Parameters
    ----------
    telescope : TelescopeParameters
        The telescope parameters for the instrument.
    bandpass : BandpassParameters
        The bandpass parameters for the instrument.
    detector : DetectorParameters
        The detector parameters for the instrument.

    Attributes
    ----------
    telescope : TelescopeParameters
        The telescope parameters for the instrument.
    bandpass : BandpassParameters
        The bandpass parameters for the instrument.
    detector : DetectorParameters
        The detector parameters for the instrument.

    """

    def __init__(
        self,
        telescope: Union[SingleDishParameters,CoronagraphParameters],
        bandpass: BandpassParameters,
        detector: DetectorParameters
    ):
        self.telescope = telescope
        self.bandpass = bandpass
        self.detector = detector
    @classmethod
    def _from_dict(cls, d: dict):
        if 'coronagraph' in d.keys():
            telescope = CoronagraphParameters.from_dict(d['coronagraph'])
        elif 'single' in d.keys():
            telescope = SingleDishParameters.from_dict(d['single'])
        else:
            raise KeyError('Cannot find Telescope Parameters key')
        return cls(
            telescope = telescope,
            bandpass = BandpassParameters.from_dict(d['bandpass']),
            detector = DetectorParameters.from_dict(d['detector'])
        )
    def to_psg(self):
        """
        Convert the instrument parameters to the PSG input format.

        Returns
        -------
        dict
            A dictionary representing the instrument parameters in the PSG input format.

        """
        config = {
        }
        config.update(self.telescope.to_psg())
        config.update(self.bandpass.to_psg())
        config.update(self.detector.to_psg())
        return config
    @classmethod
    def mirecle(cls):
        """
        2m MIRECLE setup [1].
        Returns an InstrumentParameters instance initialized with the 2m MIRECLE setup.

        Returns
        -------
        InstrumentParameters
            `InstrumentParameters` instance for the 2m MIRECLE setup.

        References
        ----------
        [1] :cite:t:`2022AJ....164..176M`
        """
        return cls(
            telescope=SingleDishParameters.mirecle(),
            bandpass=BandpassParameters.mirecle(),
            detector=DetectorParameters.mirecle()
        )
    @classmethod
    def miri_lrs(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['instrument']['miri-lrs'])
    @classmethod
    def niriss_soss(cls):
        path = PRESET_PATH / 'jwst.yaml'
        with open(path, 'r',encoding='UTF-8') as file:
            data = yaml.safe_load(file)
            return cls.from_dict(data['instrument']['niriss-soss'])
