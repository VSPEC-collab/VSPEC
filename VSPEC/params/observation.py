"""
Observation parameters
"""
from astropy import units as u
from VSPEC.config import flux_unit as default_flux_unit
from VSPEC.params.base import BaseParameters


class ObservationParameters(BaseParameters):
    """
    Class storing parameters for observations.

    Parameters
    ----------
    observation_time : astropy.units.Quantity
        The total duration of the observation.
    integration_time : astropy.units.Quantity
        The integration time of each epoch of observation.
    zodi : float
        The level of the zodiacal background. From 
        PSG handbook: '(1.0:Ecliptic pole/minimum, 
        2.0:HST/JWST low values, 10.0:Normal values,
        100.0:Close to ecliptic/Sun)'

    Attributes
    ----------
    observation_time : astropy.units.Quantity
        The total duration of the observation.
    integration_time : astropy.units.Quantity
        The integration time of each epoch of observation.
    zodi : float
        The level of the zodiacal background. From 
        PSG handbook: '(1.0:Ecliptic pole/minimum, 
        2.0:HST/JWST low values, 10.0:Normal values,
        100.0:Close to ecliptic/Sun)'

    Raises
    ------
    ValueError
        If the integration time is longer than the total observation time.

    Properties
    ----------
    total_images : int
        The total number of images based on the observation time and integration time.

    """

    def __init__(
        self,
        observation_time: u.Quantity,
        integration_time: u.Quantity,
        zodi: float
    ):
        self.observation_time = observation_time
        self.integration_time = integration_time
        self.zodi = zodi
        self._validate()

    def _validate(self):
        if self.integration_time > self.observation_time:
            raise ValueError(
                'Length of integrations cannot be longer than the total observation.')
        if self.zodi < 1.0:
            raise ValueError(
                'Zodi background must be >= 1.0'
            )

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
            zodi = float(d['zodi'])
        )
    def to_psg(self):
        """
        Convert the observation parameters to the PSG input format.

        Returns
        -------
        dict
            A dictionary representing the observation parameters in the PSG input format.

        """
        return {
            'GEOMETRY': 'Observatory',
            'GENERATOR-TELESCOPE2': f'{self.zodi}'
        }


class BandpassParameters(BaseParameters):
    """
    Class to store bandpass parameters for observations.

    Parameters:
    -----------
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

    Class Methods:
    --------------
    mirecle(cls)
        Returns a BandpassParameters instance initialized with
        the MIRECLE setup :cite:p:`2022AJ....164..176M`.

    Attributes:
    -----------
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

        Returns:
        --------
        BandpassParameters:
            `BandpassParameters` instance with the MIRECLE setup.

        References:
        -----------
        [1] :cite:t:`2022AJ....164..176M`
        """
        return cls(
            1*u.um,
            18*u.um,
            50,
            u.um,
            default_flux_unit
        )


class ccdParameters(BaseParameters):
    """
    Class to store CCD parameters for observations.

    Parameters:
    -----------
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

    Class Methods:
    --------------
    mirecle(cls)
        Returns a CCDParameters instance initialized with the MIRECLE setup.

    Attributes:
    -----------
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
            pixel_sampling = int(d['pixel_sampling']),
            read_noise = u.Quantity(d['read_noise']),
            dark_current = u.Quantity(d['dark_current']),
            throughput = float(d['throughput']),
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
            'GENERATOR-NOISE1': f'{self.read_noise.to_value(u.electron):.1f}',
            'GENERATOR-NOISE2': f'{self.dark_current.to_value(u.electron/u.s):.1f}',
            'GENERATOR-NOISEOEFF': f'{self.throughput:.2f}',
            'GENERATOR-NOISEOEMIS': f'{self.emissivity:.2f}',
            'GENERATOR-NOISEOTEMP': f'{self.temperature.to_value(u.K):.1f}'
        }

    @classmethod
    def mirecle(cls):
        """
        MIRECLE setup [1].
        Returns a CCDParameters instance initialized with the MIRECLE setup.

        Returns:
        --------
        CCDParameters:
            CCDParameters instance for the MIRECLE setup.

        References:
        -----------
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


class DetectorParameters(BaseParameters):
    """
    Class to store detector parameters for observations.

    Parameters:
    -----------
    beam_width : astropy.units.Quantity
        The beam width of the detector.
    integration_time : astropy.units.Quantity
        The integration time of the detector.
    ccd : ccdParameters
        The CCD parameters for the detector.

    Class Methods:
    --------------
    mirecle(cls)
        Returns a DetectorParameters instance initialized with the MIRECLE setup.

    Attributes:
    -----------
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
    def mirecle(cls):
        """
        MIRECLE setup [1].
        Returns a DetectorParameters instance initialized with the MIRECLE setup.

        Returns:
        --------
        DetectorParameters:
            DetectorParameters instance for the MIRECLE setup.

        References:
        -----------
        [1] :cite:t:`2022AJ....164..176M`
        """
        return cls(
            beam_width=5*u.arcsec,
            integration_time=0.5*u.s,
            ccd=ccdParameters.mirecle()
        )


class InstrumentParameters(BaseParameters):
    """
    Class to store instrument parameters for observations.

    Parameters:
    -----------
    aperture : astropy.units.Quantity
        The aperture size of the instrument.
    bandpass : BandpassParameters
        The bandpass parameters for the instrument.
    detector : DetectorParameters
        The detector parameters for the instrument.

    Class Methods:
    --------------
    mirecle(cls)
        Returns an `InstrumentParameters` instance initialized with the 2m MIRECLE setup.

    Attributes:
    -----------
    aperture : astropy.units.Quantity
        The aperture size of the instrument.
    bandpass : BandpassParameters
        The bandpass parameters for the instrument.
    detector : DetectorParameters
        The detector parameters for the instrument.

    """

    def __init__(
        self,
        aperture: u.Quantity,
        bandpass: BandpassParameters,
        detector: DetectorParameters
    ):
        self.aperture = aperture
        self.bandpass = bandpass
        self.detector = detector
    @classmethod
    def _from_dict(cls, d: dict):
        return cls(
            aperture = u.Quantity(d['aperture']),
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
            'GENERATOR-DIAMTELE': f'{self.aperture.to_value(u.m):.2f}',
            'GENERATOR-TELESCOPE': 'SINGLE'
        }
        config.update(self.bandpass.to_psg())
        config.update(self.detector.to_psg())
        return config
    @classmethod
    def mirecle(cls):
        """
        2m MIRECLE setup [1].
        Returns an InstrumentParameters instance initialized with the 2m MIRECLE setup.

        Returns:
        --------
        InstrumentParameters:
            `InstrumentParameters` instance for the 2m MIRECLE setup.

        References:
        -----------
        [1] :cite:t:`2022AJ....164..176M`
        """
        return cls(
            aperture=2*u.m,
            bandpass=BandpassParameters.mirecle(),
            detector=DetectorParameters.mirecle()
        )
