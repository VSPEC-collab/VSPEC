"""VSPEC star

This module contains the code to govern the
behavior of a variable star.

The `VSPEC` star model is designed modularly to allow
for both simple and complex behaviors. Currently, it
is represended by a rectangular grid of points on the stellar
surface, each assigned an effective temperature. At any given
time, the model computes the surface coverage fractions of
each temperature visible to the observer, accounting for the
shperical geometry, limb darkening, and any occultation
from a transiting planet.

Once the surface coverage is calculated, a composite spectrum
is computed from a grid of PHOENIX stellar spectra :cite:p:`2013A&A...553A...6H`.
As of `VSPEC 0.1`, we have spectra between 2300 K and 3900 K, with steps of
100 K. Each spectrum has :math:`\log{g} = 5.0` and solar metalicity.
The raw spectra span from 0.1 to 20 um with 5e-6 um steps, however binned
versions are available for faster runtimes.

The attributes of the `Star` class describe the bulk properties of the star,
including radius, period, and the effective temperature of quiet photosphere.
Herein we refer to this temperature as the photosphere temperature to differentiate
it from the temperature of spots, faculae, or other sources of variability
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.units.quantity import Quantity
from typing import Tuple

from VSPEC.helpers import CoordinateGrid
from VSPEC.helpers import MSH, get_angle_between, proj_ortho, calc_circ_fraction_inside_unit_circle
from VSPEC.variable_star_model.spots import SpotCollection, SpotGenerator
from VSPEC.variable_star_model.faculae import FaculaCollection, FaculaGenerator, Facula
from VSPEC.variable_star_model.flares import FlareCollection, FlareGenerator
from VSPEC.variable_star_model.granules import Granulation


class Star:
    """
    Star object representing a variable star.

    Parameters
    ----------
    Teff : astropy.units.Quantity
        Effective temperature of the stellar photosphere.
    radius : astropy.units.Quantity
        Stellar radius.
    period : astropy.units.Quantity
        Stellar rotational period.
    spots : SpotCollection
        Initial spots on the stellar surface.
    faculae : FaculaCollection
        Initial faculae on the stellar surface.
    distance : astropy.units.Quantity , default=1*u.pc
        Distance to the star.
    Nlat : int, default=500
        The number of latitude points on the stellar sufrace.
    Nlon : int, default=1000
        The number of longitude points on the stellar surface.
    gridmaker : CoordinateGrid, default=None
        A `CoordinateGrid` object to create the stellar sufrace grid.
    flare_generator : FlareGenerator, default=None
        Flare generator object.
    spot_generator : SpotGenerator, default=None
        Spot generator object.
    fac_generator : FaculaGenerator, default=None
        Facula generator object.
    ld_params : list, default=[0, 1, 0]
        Limb-darkening parameters.

    Attributes
    ----------
    Teff : astropy.units.Quantity
        Effective temperature of the stellar photosphere.
    radius : astropy.units.Quantity
        Stellar radius.
    distance : astropy.units.Quantity
        Distance to the star.
    period : astropy.units.Quantity
        Stellar rotational period.
    spots : SpotCollection
        Spots on the stellar surface.
    faculae : FaculaCollection
        Faculae on the stellar surface.
    gridmaker : CoordinateGrid
        Object to create the coordinate grid of the surface.
    map : astropy.units.Quantity
        Pixel map of the stellar surface.
    flare_generator : FlareGenerator
        Flare generator object.
    spot_generator : SpotGenerator
        Spot generator object.
    fac_generator : FaculaGenerator
        Facula generator object.
    u1 : float
        Limb-darkening parameter u1.
    u2 : float
        Limb-darkening parameter u2.
    """
    def __init__(self, Teff: u.Quantity,
                 radius: u.Quantity,
                 period: u.Quantity,
                 spots: SpotCollection,
                 faculae: FaculaCollection,
                 distance: u.Quantity = 1*u.pc,
                 Nlat: int = 500,
                 Nlon: int = 1000,
                 gridmaker: CoordinateGrid = None,
                 flare_generator: FlareGenerator = None,
                 spot_generator: SpotGenerator = None,
                 fac_generator: FaculaGenerator = None,
                 granulation: Granulation = None,
                 u1: float = 1,
                 u2: float = 0
    ):
        self.Teff = Teff
        self.radius = radius
        self.distance = distance
        self.period = period
        self.spots = spots
        self.faculae = faculae
        if not gridmaker:
            self.gridmaker = CoordinateGrid(Nlat, Nlon)
        else:
            self.gridmaker = gridmaker
        self.faculae.gridmaker = self.gridmaker
        self.spots.gridmaker = self.gridmaker

        if flare_generator is None:
            self.flare_generator = FlareGenerator(self.Teff, self.period)
        else:
            self.flare_generator = flare_generator

        if spot_generator is None:
            self.spot_generator = SpotGenerator(500*MSH, 200*MSH, umbra_teff=self.Teff*0.75,
                                                penumbra_teff=self.Teff*0.85, Nlon=Nlon, Nlat=Nlat, gridmaker=self.gridmaker)
        else:
            self.spot_generator = spot_generator

        if fac_generator is None:
            self.fac_generator = FaculaGenerator(
                R_peak=300*u.km, R_HWHM=100*u.km, Nlon=Nlon, Nlat=Nlat)
        else:
            self.fac_generator = fac_generator
        self.granulation = granulation
        self.u1 = u1
        self.u2 = u2
        self.set_spot_grid()
        self.set_fac_grid()

    def set_spot_grid(self):
        for spot in self.spots.spots:
            spot.set_gridmaker(self.gridmaker)
    def set_fac_grid(self):
        for fac in self.faculae.faculae:
            fac.set_gridmaker(self.gridmaker)

    @property
    def map(self):
        """
        Create a map of the stellar surface based on spots.

        Returns
        -------
        pixelmap : astropy.units.Quantity , Shape(self.gridmaker.Nlon,self.gridmaker.Nlat)
            Map of stellar surface with effective temperature assigned to each pixel.
        """
        return self.spots.map_pixels(self.radius, self.Teff)

    def age(self, time):
        """
        Age the spots and faculae on the stellar surface according
        to their own `age` methods. Remove the spots that have decayed.

        Parameters
        ----------
        time : astropy.units.Quantity 
            Length of time to age the features on the stellar surface.
            For most realistic behavior, `time` should be much less than
            spot or faculae lifetime.
        """
        self.spots.age(time)
        self.faculae.age(time)

    def add_spot(self, spot):
        """
        Add one or more spots to the stellar surface.

        Parameters
        ----------
        spot : StarSpot or sequence of StarSpot
            The `StarSpot` object(s) to add.
        """
        self.spots.add_spot(spot)

    def add_fac(self, facula):
        """
        Add one or more faculae to the stellar surface.

        Parameters
        ----------
        facula : Facula or sequence of Facula
            The Facula object(s) to add.

        """
        self.faculae.add_faculae(facula)

    def get_mu(self,lat0:u.Quantity,lon0:u.Quantity):
        """
        Get the cosine of the angle from disk center.

        Parameters
        ----------
        lat0 : astropy.units.Quantity
            The sub-observer latitude.
        lon0 : astropy.units.Quantity
            The sub-observer longitude
        
        Returns
        -------
        mu : np.ndarray
            An array of cos(x) where x is
            the angle from disk center.
        
        Notes
        -----
        Recall
        ..math:
            \mu = cos(x)
        """
        latgrid, longrid = self.gridmaker.grid()
        mu = (np.sin(lat0) * np.sin(latgrid)
                 + np.cos(lat0) * np.cos(latgrid)
                 * np.cos(lon0-longrid))
        return mu

    def ld_mask(self,mu)->np.ndarray:
        """
        Get a translucent mask based on limb darkeining parameters.

        Parameters
        ----------
        mu : np.ndarray
            The cosine of the angle from disk center

        Returns
        -------
        mask : np.ndarray
            The limb-darkened mask.
        """
        mask = 1 - self.u1 * (1 - mu) - self.u2 * (1 - mu)**2
        behind_star = mu<0.
        mask[behind_star] = 0
        return mask
    
    def get_jacobian(self)->np.ndarray:
        """
        Get the relative area of each point.

        Returns
        -------
        jacobian : np.ndarray
            The area of each point
        """
        latgrid, _ = self.gridmaker.grid()
        jacobian = np.sin(latgrid + 90*u.deg)
        return jacobian
    
    def add_faculae_to_map(
        self,
        lat0:u.Quantity,
        lon0:u.Quantity
    ):
        """
        Add the faculae to the surface map.

        Parameters
        ----------
        lat0 : astropy.units.Quantity
            The sub-observer latitude.
        lon0 : astropy.units.Quantity
            The sub-observer longitude.
        
        Returns
        -------
        teffmap : astropy.units.Quantity
            A temperature map of the surface
        """
        map_from_spots = self.map
        mu = self.get_mu(lat0,lon0)
        faculae:Tuple[Facula] = self.faculae.faculae
        for facula in faculae:
            angle = get_angle_between(lat0,lon0,facula.lat,facula.lon)
            inside_fac = facula.map_pixels(self.radius)
            fracs = facula.fractional_effective_area(angle)
            teff_wall, teff_floor = fracs.keys()
            frac = fracs[teff_wall].value
            mu_of_fac_pix = mu[inside_fac]
            border_mu = np.percentile(mu_of_fac_pix,100*frac)
            wall_pix = inside_fac & (mu <= border_mu)
            floor_pix = inside_fac & (mu > border_mu)
            map_from_spots[wall_pix] = teff_wall
            map_from_spots[floor_pix] = teff_floor
        return map_from_spots
    
    def get_pl_frac(
        self,
        angle_past_midtransit:u.Quantity,
        orbit_radius:u.Quantity,
        planet_radius:u.Quantity,
        inclination:u.Quantity
    ):
        """
        Get planet fraction

        Parameters
        ----------
        angle_past_midtransit : astropy.units.Quantity
            The phase of the planet past the 180 degree mid transit point.
        orbit_radius : astropy.units.Quantity
            The radius of the planet's orbit.
        radius : astropy.units.Quantity
            The radius of the planet.
        
        inclination : astropy.units.Quantity
            The inclination of the planet. 90 degrees is transiting.
        """
        x = (orbit_radius/self.radius * np.sin(angle_past_midtransit)).to_value(u.dimensionless_unscaled)
        y = (orbit_radius/self.radius * np.cos(angle_past_midtransit) * np.cos(inclination)).to_value(u.dimensionless_unscaled)
        rad = (planet_radius/self.radius).to_value(u.dimensionless_unscaled)
        return 1-calc_circ_fraction_inside_unit_circle(x,y,rad)

    def get_transit_mask(
        self,
        lat0:u.Quantity,
        lon0:u.Quantity,
        orbit_radius:u.Quantity,
        radius:u.Quantity,
        phase:u.Quantity,
        inclination:u.Quantity
    ):
        """
        Get a mask describing which pixels are covered by a transiting planet.

        Parameters
        ----------
        lat0 : astropy.units.Quantity
            The sub-observer latitude.
        lon0 : astropy.units.Quantity
            The sub-observer longitude.
        orbit_radius : astropy.units.Quantity
            The radius of the planet's orbit.
        radius : astropy.units.Quantity
            The radius of the planet.
        phase : astropy.units.Quantity
            The phase of the planet. 180 degrees is mid transit.
        inclination : astropy.units.Quantity
            The inclination of the planet. 90 degrees is transiting.
        """
        eclipse = False
        if np.cos(phase) > 0:
            eclipse = True
        angle_past_midtransit = phase - 180*u.deg
        x = (orbit_radius/self.radius * np.sin(angle_past_midtransit)).to_value(u.dimensionless_unscaled)
        y = (orbit_radius/self.radius * np.cos(angle_past_midtransit) * np.cos(inclination)).to_value(u.dimensionless_unscaled)
        rad = (radius/self.radius).to_value(u.dimensionless_unscaled)
        if np.sqrt(x**2 + y**2) > 1 + 2*rad: # no transit
            return self.gridmaker.zeros().astype('bool'), 1.0
        elif eclipse:
            planet_fraction = self.get_pl_frac(angle_past_midtransit,orbit_radius,radius,inclination)
            return self.gridmaker.zeros().astype('bool'), planet_fraction
        else:
            llat,llon = self.gridmaker.grid()
            xcoord,ycoord = proj_ortho(lat0,lon0,llat,llon)
            rad_map = np.sqrt((xcoord-x)**2 + (ycoord-y)**2)
            covered = np.where(rad_map<=rad,1,0).astype('bool')
            return covered,1.0


    def calc_coverage(
        self,
        sub_obs_coords:dict,
        granulation_fraction:float=0.0,
        orbit_radius:u.Quantity = 1*u.AU,
        planet_radius:u.Quantity = 1*u.R_earth,
        phase:u.Quantity = 90*u.deg,
        inclination:u.Quantity = 0*u.deg

    ):
        """
        Calculate coverage

        Calculate coverage fractions of various Teffs on stellar surface
        given coordinates of the sub-observation point.

        Parameters
        ----------
        sub_obs_coord : dict
            A dictionary giving coordinates of the sub-observation point.
            This is the point that is at the center of the stellar disk from the view of
            an observer. Format: {'lat':lat,'lon':lon} where lat and lon are
            `astropy.units.Quantity` objects.
        granulation_fraction : float
            The fraction of the quiet photosphere that has a lower Teff due to granulation

        Returns
        -------
        total_data : dict
            Dictionary with Keys as Teff quantities and Values as surface fraction floats.
        covered_data : dict
            Dictionary with Keys as Teff quantities and Values as surface fraction floats covered
            by a transiting planet.
        pl_frac : float
            The fraction of the planet that is visble. This is in case of an eclipse.
        """
        cos_c = self.get_mu(sub_obs_coords['lat'],sub_obs_coords['lon'])
        ld = self.ld_mask(cos_c)
        jacobian = self.get_jacobian()

        surface_map = self.add_faculae_to_map(sub_obs_coords['lat'],sub_obs_coords['lon'])
        covered, pl_frac = self.get_transit_mask(
            sub_obs_coords['lat'],sub_obs_coords['lon'],
            orbit_radius=orbit_radius,
            radius=planet_radius,
            phase=phase,
            inclination=inclination
        )

        Teffs = np.unique(surface_map)
        total_data = {}
        covered_data = {}
        total_area = np.sum(ld*jacobian)
        for teff in Teffs:
            pix_has_teff = np.where(surface_map==teff,1,0)
            nominal_area = np.sum(pix_has_teff*ld*jacobian)
            covered_area = np.sum(pix_has_teff*ld*jacobian*(covered))
            total_data[teff] = (nominal_area/total_area).to_value(u.dimensionless_unscaled)
            covered_data[teff] = (covered_area/total_area).to_value(u.dimensionless_unscaled)
        granulation_teff = self.Teff - self.granulation.dteff
        if granulation_teff not in Teffs: # initialize. This way it's okay if there's something else with that Teff too.
            total_data[granulation_teff] = 0
            covered_data[granulation_teff] = 0
        
        phot_frac = total_data[self.Teff]
        total_data[self.Teff] =  phot_frac * (1-granulation_fraction)
        total_data[granulation_teff] += phot_frac * granulation_fraction
        phot_frac = covered_data[self.Teff]
        covered_data[self.Teff] =  phot_frac * (1-granulation_fraction)
        covered_data[granulation_teff] += phot_frac * granulation_fraction

        return total_data,covered_data,pl_frac

    def calc_orthographic_mask(self, sub_obs_coords):
        """
        Calculate orthographic mask.

        Get the value of the orthographic mask at each point on the stellar surface when
        viewed from the specified sub-observation point. 


        Parameters
        ----------
        sub_obs_coords : dict
            A dictionary containing coordinates of the sub-observation point. This is the 
            point that is at the center of the stellar disk from the view of an observer. 
            Format: {'lat':lat,'lon':lon} where lat and lon are `astropy.units.Quantity` objects.

        Returns
        -------
        numpy.ndarray
            The effective pixel size when projected onto an orthographic map.
        """

        latgrid, longrid = self.gridmaker.grid()
        cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(latgrid)
                 + np.cos(sub_obs_coords['lat']) * np.cos(latgrid)
                 * np.cos(sub_obs_coords['lon']-longrid))
        ld = self.ld_mask(cos_c)
        return ld

    def birth_spots(self, time):
        """
        Create new spots from a spot generator.

        Parameters
        ----------
        time : astropy.units.Quantity 
            Time over which these spots should be created.

        """
        self.spots.add_spot(self.spot_generator.birth_spots(time, self.radius))
        
    def birth_faculae(self, time):
        """
        Create new faculae from a facula generator.

        Parameters
        ----------
        time : astropy.units.Quantity 
            Time over which these faculae should be created.


        """
        self.faculae.add_faculae(
            self.fac_generator.birth_faculae(time, self.radius, self.Teff))

    def average_teff(self, sub_obs_coords):
        """
        Calculate the average Teff of the star given a sub-observation point
        using the Stephan-Boltzman law. This can approximate a lightcurve for testing.

        Parameters
        ----------
        sub_obs_coords : dict
            A dictionary containing coordinates of the sub-observation point. This is the 
            point that is at the center of the stellar disk from the view of an observer. 
            Format: {'lat':lat,'lon':lon} where lat and lon are `astropy.units.Quantity` objects.

        Returns
        -------
        astropy.units.Quantity 
            Bolometric average Teff of stellar disk.

        """
        dat,_,_ = self.calc_coverage(sub_obs_coords)
        num = 0
        den = 0
        for teff in dat.keys():
            num += teff**4 * dat[teff]
            den += dat[teff]
        return ((num/den)**(0.25)).to(u.K)

    def plot_spots(self, view_angle, sub_obs_point=None):
        """
        Plot spots on a map using the orthographic projection.

        Parameters
        ----------
        view_angle: dict
            Dictionary with two keys, 'lon' and 'lat', representing the longitude and
            latitude of the center of the projection in degrees.
        sub_obs_point: tuple, default=None
            Tuple with two elements, representing the longitude and latitude of the
            sub-observer point in degrees. If provided, a gray overlay is plotted
            indicating the regions that are visible from the sub-observer point.

        Returns
        -------
        fig: matplotlib.figure.Figure
            The resulting figure object.

        Notes
        -----
        This method uses the numpy and matplotlib libraries to plot a map of the spots
        on the stellar surface using an orthographic projection centered at the
        coordinates provided in the `view_angle` parameter. The pixel map is obtained
        using the `get_pixelmap` method of `Star`. If the `sub_obs_point` parameter
        is provided, a gray overlay is plotted indicating the visible regions from the
        sub-observer point.
        """
        # This makes cartopy and optional dependency
        import cartopy.crs as ccrs

        pmap = self.add_faculae_to_map(view_angle['lat'],view_angle['lon']).value
        lat,lon = self.gridmaker.oned()
        proj = ccrs.Orthographic(
            central_longitude=view_angle['lon'], central_latitude=view_angle['lat'])
        fig = plt.figure(figsize=(5, 5), dpi=100, frameon=False)
        ax = plt.axes(projection=proj, fc="r")
        ax.outline_patch.set_linewidth(0.0)
        ax.imshow(lat,lon,pmap.T,
            transform=ccrs.PlateCarree(),
            cmap='viridis',

        )
        if sub_obs_point is not None:
            mask = self.calc_orthographic_mask(sub_obs_point)
            ax.imshow(lat,lon,
                mask.T,
                transform=ccrs.PlateCarree(),
                cmap='gray',
                alpha=0.7
            )
        return fig

    def plot_faculae(self, view_angle):
        """
        Plot faculae on a map using orthographic projection.

        Parameters
        ----------
        view_angle: dict
            Dictionary with two keys, 'lon' and 'lat', representing the longitude and
            latitude of the center of the projection in degrees.

        Returns
        -------
        fig: matplotlib.figure.Figure
            The resulting figure object.

        Notes
        -----
        This method uses the numpy and matplotlib libraries to plot a map of the faculae
        on the stellar surface using an orthographic projection centered at the
        coordinates provided in the `view_angle` parameter. The faculae are obtained from
        the `Star`'s faculae attribute and are mapped onto pixels using the `map_pixels`
        method. The resulting map is plotted using an intensity map with faculae pixels
        represented by the value 1 and non-faculae pixels represented by the value 0.
        """
        # This makes cartopy and optional dependency
        import cartopy.crs as ccrs

        int_map, map_keys = self.faculae.map_pixels(
            self.map, self.radius, self.Teff)
        is_fac = ~(int_map == 0)
        int_map[is_fac] = 1
        proj = ccrs.Orthographic(
            central_longitude=view_angle['lon'], central_latitude=view_angle['lat'])
        fig = plt.figure(figsize=(5, 5), dpi=100, frameon=False)
        ax = plt.axes(projection=proj, fc="r")
        ax.outline_patch.set_linewidth(0.0)
        ax.imshow(
            int_map.T,
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[0, 360, -90, 90],
            interpolation="none",
            regrid_shape=(self.gridmaker.Nlat, self.gridmaker.Nlon)
        )
        return fig

    def get_flares_over_observation(self, time_duration: Quantity[u.hr]):
        """
        Generate a collection of flares over a specified observation period.

        Parameters
        ----------
        time_duration: astropy.units.Quantity 
            The duration of the observation period.

        Notes
        -----
        This method uses the `FlareGenerator` attribute of the `Star` object to generate
        a distribution of flare energies, and then generates a series of flares over the
        specified observation period using these energies. The resulting collection of
        flares is stored in the `Star`'s `flares` attribute.
        """
        energy_dist = self.flare_generator.generage_E_dist()
        flares = self.flare_generator.generate_flare_series(
            energy_dist, time_duration)
        self.flares = FlareCollection(flares)

    def get_flare_int_over_timeperiod(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr], sub_obs_coords):
        """
        Compute the total flare integral over a specified time period and sub-observer point.

        Parameters
        ----------
        tstart: astropy.units.Quantity 
            The start time of the period.
        tfinish: astropy.units.Quantity 
            The end time of the period.
        sub_obs_coords : dict
            A dictionary containing coordinates of the sub-observation point. This is the 
            point that is at the center of the stellar disk from the view of an observer. 
            Format: {'lat':lat,'lon':lon} where lat and lon are `astropy.units.Quantity` objects.


        Returns
        -------
        flare_timeareas: list of dict
            List of dictionaries containing flare temperatures and integrated
            time-areas. In the format [{'Teff':9000*u.K,'timearea'=3000*u.Unit('km2 hr)},...]

        Notes
        -----
        This method computes the total flare integral over each flare in the `flares` 
        attribute of the `Star` object that falls within the specified time period and is visible
        from the sub-observer point defined by `sub_obs_coords`. The result is returned
        as a list of dictionaries representing the teff and total flare integral of each flare.
        """
        flare_timeareas = self.flares.get_flare_integral_in_timeperiod(
            tstart, tfinish, sub_obs_coords)
        return flare_timeareas

    def generate_mature_spots(self, coverage: float):
        """
        Generate new mature spots with a specified coverage.

        Parameters
        ----------
        coverage: float
            The coverage of the new spots.

        Notes
        -----
        This method uses the `SpotGenerator` attribute of the current object to generate a
        set of new mature spots with the specified coverage. The new spots are added to 
        the object's `spots` attribute and the pixel map is updated using the new spots.
        """
        new_spots = self.spot_generator.generate_mature_spots(
            coverage, self.radius)
        self.spots.add_spot(new_spots)
    

    def get_granulation_coverage(self,time:u.Quantity)->np.ndarray:
        """
        Calculate the coverage by granulation at each point in `time`.

        Parameters
        ----------
        time : astropy.units.Quantity
            The points on the time axis.
        
        Returns
        -------
        np.ndarray
            The coverage corresponding to each `time` point.
        """
        if self.granulation is None:
            return np.zeros(shape=time.shape)
        else:
            coverage = self.granulation.get_coverage(time)
            return np.where(np.isnan(coverage),0,coverage)
