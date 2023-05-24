"""
Observation parameters
"""
from astropy import units as u
from VSPEC.config import flux_unit as default_flux_unit


class ObservationParameters:
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
    observation_time : astropy.units.Quantity
        The total duration of the observation.
    integration_time : astropy.units.Quantity
        The integration time of each epoch of observation.

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


class BandpassParameters:
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

    def __init__(
        self,
        wl_blue: u.Quantity,
        wl_red: u.Quantity,
        resolving_power: int,
        wavelength_unit: u.Unit,
        flux_unit: u.Unit,
    ):
        self.wl_blue = wl_blue
        self.wl_red = wl_red
        self.resolving_power = resolving_power
        self.wavelength_unit = wavelength_unit
        self.flux_unit = flux_unit

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


class ccdParameters:
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


class DetectorParameters:
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


class InstrumentParameters:
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
