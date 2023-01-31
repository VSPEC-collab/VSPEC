import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u, constants as c
from astropy.units.quantity import Quantity
import cartopy.crs as ccrs
from xoflares.xoflares import _flareintegralnp as flareintegral, eval_get_light_curve,get_light_curvenp
from VSPEC.helpers import to_float
from copy import deepcopy
from typing import List
import typing as Typing

MSH = u.def_unit('micro solar hemisphere', 1e-6 * 0.5 * 4*np.pi*u.R_sun**2)

class StarSpot:
    """
    Class to govern behavior of spots on a star's surface

    Args:
        lat (astropy.units.quantity.Quantity [angle]): latitude of spot center.
            Can be in radians or degrees from equator. North is positive.
        lon (astropy.units.quantity.Quantity [angle]): longitude of spot center.
            Can be in radians or degrees from prime meridian. East is positive.
        Amax (astropy.units.quantity.Quantity [area]): Area spot reaches before it decays.
        A0 (astropy.units.quantity.Quantity [area]): Current spot area.
        Teff_umbra (astropy.units.quantity.Quantity [temperature]): Effective temperature of spot umbra.
        Teff_penumbra (astropy.units.quantity.Quantity [temperature]): Effective temperature of spot penumbra.
        r_A (float): Ratio of total spot area to umbra area. From 2013PhDT.......359G = 5+/-1 (compiled from various sources)
        growing (bool): Whether or not the spot is growing.
        growth_rate (astropy.units.quantity.Quantity [1/time]): Fractional growth of the spot for a given unit time.
            From 2013PhDT.......359G can be 0.52/day to 1.83/day (compiled form various sources)
        Nlat (int): number of latitude points. Default 500
        Nlon (int): number of longitude points Default 1000
        gridmake (CoordinateGrid): grid maker object. Default None

    Returns:
        None
    """
    def __init__(
        self,lat:Quantity[u.deg],lon:Quantity[u.deg],Amax:Quantity[MSH],A0:Quantity[MSH],
        Teff_umbra:Quantity[u.K],Teff_penumbra:Quantity[u.K],r_A:float=5,growing:bool=True,
        growth_rate:Quantity[1/u.day] = 0.52/u.day,decay_rate:Quantity[MSH/u.day] = 10.89 * MSH/u.day,
        Nlat:int =500,Nlon:int =1000,gridmaker=None
        ):
        
        self.coords = {'lat':lat,'lon':lon}
        self.area_max = Amax
        self.area_current = A0
        self.Teff_umbra = Teff_umbra
        self.Teff_penumbra = Teff_penumbra
        self.decay_rate = decay_rate
        self.total_area_over_umbra_area = r_A
        self.is_growing = growing
        self.growth_rate = growth_rate

        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat,Nlon)
        else:
            self.gridmaker = gridmaker
        latgrid,longrid = self.gridmaker.grid()
        lat0 = self.coords['lat']
        lon0 = self.coords['lon']
        self.r = 2* np.arcsin(np.sqrt(np.sin(0.5*(lat0-latgrid))**2
                         + np.cos(latgrid)*np.cos(lat0)*np.sin(0.5*(lon0 - longrid))**2))
    def radius(self)->Quantity[u.km]:
        """radius

        Get the radius of the spot.

        Args:
            None
        
        Returns:
            (astropy.units.quantity.Quantity [length]): Radius of spot.
        """
        return np.sqrt(self.area_current/np.pi).to(u.km)
    def angular_radius(self,star_rad:Quantity[u.R_sun])->Quantity[u.deg]:
        """angular radius

        Get the angular radius of the spot

        Args:
            star_rad (astropy.units.quantity.Quantity [length]): radius of the star.
        
        Returns:
            (astropy.units.quantity.Quantity [angle]): angular radius of the spot
        """
        cos_angle = 1 - self.area_current/(2*np.pi*star_rad**2)
        return (np.arccos(cos_angle)).to(u.deg)
    
    def map_pixels(self,star_rad:Quantity[u.R_sun])->dict:
        """map pixels

        Map latitude and longituide points continaing the umbra and penumbra

        Args:
            latgrid (astropy.units.quantity.Quantity [angle], shape(M,N)): grid of latitude points to map
            longrid (astropy.units.quantity.Quantity [angle], shape(M,N)): grid of longitude points to map
            star_rad (astropy.units.quantity.Quantity [length]): radius of the star.
        
        Returns:
            (dict): dictionary of points covered by the umbra and penumbra. Keys are the Teff of each region. Arrays are numpy booleans.
        """
        radius = self.angular_radius(star_rad)
        radius_umbra = radius/np.sqrt(self.total_area_over_umbra_area)
        return {self.Teff_umbra:self.r < radius_umbra,
                self.Teff_penumbra: self.r < radius}
    def surface_fraction(self,sub_obs_coords:dict,star_rad:Quantity[u.R_sun],N:int=1001)->float:
        """surface fraction

        Determine the surface fraction covered by a spot from a given angle of observation.
        This algorithm uses the orthographic projection.

        Args:
            sub_obs_coord (dict): dictionary giving coordinates of the sub-observation point.
                This is the point that is at the center of the stellar disk from the view of
                an observer. Format: {'lat':lat,'lon':lon} where lat and lon are astropy Quantity objects
            star_rad (astropy.units.quantity.Quantity [length]): radius of the star.
            N (int): number of points to use in numerical integration. N=1000 is not so different from N=100000.
        
        Returns:
            (float): fraction of observed disk covered by spot
        """
        cos_c0 = (np.sin(sub_obs_coords['lat']) * np.sin(self.coords['lat'])
                + np.cos(sub_obs_coords['lat'])* np.cos(self.coords['lat'])
                 * np.cos(sub_obs_coords['lon']-self.coords['lon']) )
        c0 = np.arccos(cos_c0)
        c = np.linspace(-90,90,N)*u.deg
        a = self.angular_radius(star_rad).to(u.deg)
        rad = a**2 - (c-c0)**2
        rad[rad<0] = 0
        integrand = 2 * np.cos(c)*np.sqrt(rad)
        return to_float(np.trapz(integrand,x=c)/(2*np.pi*u.steradian),u.Unit(''))
    
    def age(self,time:Quantity[u.s])->None:
        """age

        Age a spot according to its growth timescale and decay rate

        Args:
            time (astropy.units.quantity.Quantity [time]): length of time to age the spot.
                For most realistic behavior, time should be << spot lifetime
        
        Returns:
            None
        """
        if self.is_growing:
            tau = np.log((self.growth_rate * u.day).to(u.Unit('')) + 1)
            if tau == 0:
                time_to_max = np.inf*u.day
            else:
                time_to_max = np.log(self.area_max/self.area_current)/tau * u.day
            if time_to_max > time:
                new_area = self.area_current * np.exp(tau * time/u.day)
                self.area_current = new_area
            else:
                self.is_growing = False
                decay_time = time - time_to_max
                area_decay = decay_time * self.decay_rate
                if area_decay > self.area_max:
                    self.area_current = 0*MSH
                else:
                    self.area_current = self.area_max - area_decay
        else:
            area_decay = time * self.decay_rate
            if area_decay > self.area_max:
                self.area_current = 0*MSH
            else:
                self.area_current = self.area_current - area_decay

class SpotCollection:
    """Spot Collection

    Containter holding spots

    Args:
        *spots (StarSpot): series of StarSpot objects
        Nlat (int): number of latitude points. Default 500
        Nlon (int): number of longitude points Default 1000
        gridmake (CoordinateGrid): grid maker object. Default None
    
    Returns:
        None
    """
    def __init__(self,*spots:tuple[StarSpot],Nlat:int=500,Nlon:int=1000,gridmaker=None):
        self.spots = spots
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat,Nlon)
        else:
            self.gridmaker = gridmaker
        for spot in spots:
            spot.gridmaker = self.gridmaker
    def add_spot(self,spot:Typing.Union[StarSpot,list[StarSpot]]):
        """add spot

        Add a spot

        Args:
            spot (StarSpot or series of StarSpot): StarSpot object(s) to add
        
        Returns:
            None
        """
        if isinstance(spot, StarSpot):
            spot.gridmaker = self.gridmaker
        else:
            for s in spot:
                s.gridmaker = self.gridmaker
        self.spots += tuple(spot)
    def clean_spotlist(self):
        """clean spotlist

        Remove spots that have decayed to 0 area.

        Args:
            None
        
        Returns:
            None
        """
        spots_to_keep = []
        for spot in self.spots:
            if (spot.area_current <= 0*MSH) and (not spot.is_growing):
                pass
            else:
                spots_to_keep.append(spot)
        self.spots = spots_to_keep
    def map_pixels(self,star_rad:Quantity[u.R_sun],star_teff:Quantity[u.K]):
        """map pixels

        Map latitude and longituide points continaing the umbra and penumbra of each spot.
        For pixels with coverage from multiple spots, assign coolest Teff to that pixel.

        Args:
            latgrid (astropy.units.quantity.Quantity [angle], shape(M,N)): grid of latitude points to map
            longrid (astropy.units.quantity.Quantity [angle], shape(M,N)): grid of longitude points to map
            star_rad (astropy.units.quantity.Quantity [length]): radius of the star.
        
        Returns:
            (array of astropy.units.quantity.Quantity [temperature], Shape(M,N)): Map of stellar surface with Teff assigned to each pixel.
        """
        surface_map = self.gridmaker.zeros()*star_teff.unit + star_teff
        for spot in self.spots:
            teff_dict = spot.map_pixels(star_rad)
            #penumbra
            assign = teff_dict[spot.Teff_penumbra] & (surface_map > spot.Teff_penumbra)
            surface_map[assign] = spot.Teff_penumbra
            #umbra
            assign = teff_dict[spot.Teff_umbra] & (surface_map > spot.Teff_umbra)
            surface_map[assign] = spot.Teff_umbra
        return surface_map
    def age(self,time:Quantity[u.day]):
        """age

        Age spots according to its growth timescale and decay rate.
        Remove spots that have decayed.

        Args:
            time (astropy.units.quantity.Quantity [time]): length of time to age the spot.
                For most realistic behavior, time should be << spot lifetime
        
        Returns:
            None
        """
        for spot in self.spots:
            spot.age(time)
        self.clean_spotlist()

class Star:
    """Star
    Variable star

    Args:
        Teff (astropy.units.quantity.Quantity [temperature]): Effective temperature of stellar photosphere.
        radius (astropy.units.quantity.Quantity [length]): stellar radius
        period (astropy.units.quantity.Quantity [time]): stellar rotational period
        spots (SpotCollection): initial spots on stellar surface
        distance (astropy.units.quantity.Quantity [distance]): distance to the star, Default 1 pc
        Nlat (int): number of latitude points. Default 500
        Nlon (int): number of longitude points Default 1000
        gridmake (CoordinateGrid): grid maker object. Default None    
    Returns:
        None
    """
    def __init__(self,Teff,radius,period,spots,faculae,name='',distance = 1*u.pc,Nlat = 500,Nlon=1000,gridmaker=None,
                    flare_generator = None,spot_generator = None,fac_generator=None,ld_params = [0,1,0]):
        self.name = name
        assert isinstance(Teff,Quantity)
        self.Teff = Teff
        assert isinstance(radius,Quantity)
        self.radius = radius
        assert isinstance(distance,Quantity)
        self.distance = distance
        assert isinstance(period,Quantity)
        self.period = period
        assert isinstance(spots,SpotCollection)
        self.spots = spots
        assert isinstance(faculae,FaculaCollection)
        self.faculae = faculae
        if not gridmaker:
            self.gridmaker = CoordinateGrid(Nlat,Nlon)
        else:
            self.gridmaker = gridmaker
        self.map = self.get_pixelmap()
        self.faculae.gridmaker = self.gridmaker
        self.spots.gridmaker = self.gridmaker

        if flare_generator is None:
            self.flare_generator = FlareGenerator(self.Teff,self.period)
        else:
            self.flare_generator = flare_generator

        if spot_generator is None:
            self.spot_generator = SpotGenerator(500*MSH,200*MSH,umbra_teff=self.Teff*0.75,
            penumbra_teff=self.Teff*0.85,Nlon=Nlon,Nlat=Nlat,gridmaker=self.gridmaker)
        else:
            self.spot_generator = spot_generator

        if fac_generator is None:
            self.fac_generator = FaculaGenerator(R_peak = 300*u.km, R_HWHM = 100*u.km,Nlon=Nlon,Nlat=Nlat)
        else:
            self.fac_generator = fac_generator
        self.ld_params = ld_params
    def get_pixelmap(self):
        """get pixelmap
        Create map of stellar surface based on spots:
        
        Args:
            Nlat (int): Number of latitude points to map. Default 500
            Nlon (int): Number of longitude points to map. Default 1000
        
        Returns:
            (array of astropy.units.quantity.Quantity [temperature], Shape(Nlon,Nlat)): Map of stellar surface with Teff assigned to each pixel.

        """
        return self.spots.map_pixels(self.radius,self.Teff)
    def age(self,time):
        """age
        Age spots according to its growth timescale and decay rate.
        Remove spots that have decayed.

        Args:
            time (astropy.units.quantity.Quantity [time]): length of time to age the spot.
                For most realistic behavior, time should be << spot lifetime
        
        Returns:
            None
        """
        self.spots.age(time)
        self.faculae.age(time)
        self.map = self.get_pixelmap()
    def add_spot(self,spot):
        """add spot
        Add a spot

        Args:
            spot (StarSpot or series of StarSpot): StarSpot object(s) to add
        
        Returns:
            None
        """
        self.spots.add_spot(spot)
        self.map = self.get_pixelmap()
    def add_fac(self,facula):
        """add facula(e)
        Add a facula or faculae

        Args:
            facula (Facula or series of Faculae): Facula objects to add
        
        Returns:
            None
        """
        self.faculae.add_faculae(facula)
    def calc_coverage(self,sub_obs_coords):
        """Calculate coverage
        Calculate coverage fractions of various Teffs on stellar surface
        give coordinates of the sub-observation point.

        Args:
            sub_obs_coord (dict): dictionary giving coordinates of the sub-observation point.
                This is the point that is at the center of the stellar disk from the view of
                an observer. Format: {'lat':lat,'lon':lon} where lat and lon are astropy Quantity objects
            Nlat (int): Number of latitude points to map. Default 500
            Nlon (int): Number of longitude points to map. Default 1000
        
        Returns:
            (dict): Dictionary with Keys as Teff quantities and Values as surface fraction floats.
        """
        latgrid,longrid = self.gridmaker.grid()
        cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(latgrid)
                + np.cos(sub_obs_coords['lat'])* np.cos(latgrid)
                 * np.cos(sub_obs_coords['lon']-longrid) )
        ld = cos_c**0 * self.ld_params[0] + cos_c**1 * self.ld_params[1] + cos_c**2 * self.ld_params[2]
        ld[cos_c < 0] = 0
        jacobian = np.sin(latgrid + 90*u.deg)

        int_map, map_keys = self.faculae.map_pixels(self.map,self.radius,self.Teff)

        Teffs = np.unique(self.map)
        data = {}
        # spots and photosphere
        for teff in Teffs:
            pix = self.map == teff
            pix_sum = ((pix.astype('float32') * ld * jacobian)[int_map==0]).sum()
            data[teff] = pix_sum
        for i in map_keys.keys():
            facula = self.faculae.faculae[i]
            angle = 2 * np.arcsin(np.sqrt(np.sin(0.5*(facula.lat - sub_obs_coords['lat']))**2
                                + np.cos(facula.lat)*np.cos(sub_obs_coords['lat']) * np.sin(0.5*(facula.lon - sub_obs_coords['lon']))**2 ))
            frac_area_dict = facula.fractional_effective_area(angle)
            loc = int_map == map_keys[i]
            pix_sum = (loc.astype('float32') * ld * jacobian).sum()
            for teff in frac_area_dict.keys():
                if teff in data:
                    data[teff] = data[teff] + pix_sum * frac_area_dict[teff]
                else:
                    data[teff] = pix_sum * frac_area_dict[teff]
        total = 0
        for teff in data.keys():
            total += data[teff]
        # normalize
        for teff in data.keys():
            data[teff] = data[teff]/total
        return data
    
    def calc_orthographic_mask(self,sub_obs_coords):
        """Calculate orthographic mask
        Get value of orthographic mask at each point
        
        Args:
            sub_obs_coord (dict): dictionary giving coordinates of the sub-observation point.
                This is the point that is at the center of the stellar disk from the view of
                an observer. Format: {'lat':lat,'lon':lon} where lat and lon are astropy Quantity objects
            Nlat (int): Number of latitude points to map. Default 500
            Nlon (int): Number of longitude points to map. Default 1000
        
        Returns:
            (numpy.ndarray, Shape(Nlon,Nlat)): Effective pixel size when projected
                onto orthographic map.
        """
        
        latgrid,longrid = self.gridmaker.grid()
        cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(latgrid)
                + np.cos(sub_obs_coords['lat'])* np.cos(latgrid)
                 * np.cos(sub_obs_coords['lon']-longrid) )
        ld = cos_c**0 * self.ld_params[0] + cos_c**1 * self.ld_params[1] + cos_c**2 * self.ld_params[2]
        ld[cos_c < 0] = 0
        return ld

    def birth_spots(self,time):
        """birth spots
        Create new spots from a spot generator.

        Args:
            time (astropy.units.quantity.Quantity [time]): time over which these spots should be created.
        
        Returns:
            None
        """
        self.spots.add_spot(self.spot_generator.birth_spots(time,self.radius))
        self.map = self.get_pixelmap()
    def birth_faculae(self,time):
        """birth faculae
        Create new faculae from a facula generator.

        Args:
            time (astropy.units.quantity.Quantity [time]): time over which these faculae should be created.
        
        Returns:
            None
        """
        self.faculae.add_faculae(self.fac_generator.birth_faculae(time,self.radius,self.Teff))
    def average_teff(self,sub_obs_coords):
        """Average Teff
        Calculate the average Teff of the star given a sub-observation point
        using the Stephan-Boltzman law. This can approximate a lightcurve for testing.

        Args:
            sub_obs_coord (dict): dictionary giving coordinates of the sub-observation point.
                This is the point that is at the center of the stellar disk from the view of
                an observer. Format: {'lat':lat,'lon':lon} where lat and lon are astropy Quantity objects

        Returns:
            (astropy.units.quantity.Quantity [temperaature]): Bolometric average Teff of stellar disk
        
        """
        dat = self.calc_coverage(sub_obs_coords)
        num = 0
        den = 0
        for teff in dat.keys():
            num += teff**4 * dat[teff]
            den += dat[teff]
        return ((num/den)**(0.25)).to(u.K)


    def plot_spots(self,view_angle, sub_obs_point = None):
        pmap = self.get_pixelmap().value
        proj = ccrs.Orthographic(
                    central_longitude=view_angle['lon'], central_latitude=view_angle['lat'])
        fig = plt.figure(figsize=(5, 5), dpi=100, frameon=False)
        ax = plt.axes(projection=proj, fc="r")
        ax.outline_patch.set_linewidth(0.0)
        ax.imshow(
            pmap.T,
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[0, 360, -90, 90],
            interpolation="none",
            regrid_shape=(self.gridmaker.Nlat,self.gridmaker.Nlon)
        )
        if sub_obs_point is not None:
            mask = self.calc_orthographic_mask(sub_obs_point)
            ax.imshow(
            mask.T,
            origin="upper",
            transform=ccrs.PlateCarree(),
            extent=[0, 360, -90, 90],
            interpolation="none",
            regrid_shape=(self.gridmaker.Nlat,self.gridmaker.Nlon),
            cmap='gray',
            alpha=0.7
        )
        return fig

    def plot_faculae(self,view_angle):
        int_map, map_keys = self.faculae.map_pixels(self.map,self.radius,self.Teff)
        is_fac = ~(int_map==0)
        int_map[is_fac]=1
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
            regrid_shape=(self.gridmaker.Nlat,self.gridmaker.Nlon)
        )
        return fig
    

    def get_flares_over_observation(self,time_duration:Quantity[u.hr]):
        energy_dist = self.flare_generator.generage_E_dist()
        flares = self.flare_generator.generate_flare_series(energy_dist,time_duration)
        self.flares = FlareCollection(flares)
    
    def get_flare_int_over_timeperiod(self,tstart:Quantity[u.hr],tfinish:Quantity[u.hr],sub_obs_coords):
        flare_timeareas = self.flares.get_flare_integral_in_timeperiod(tstart,tfinish,sub_obs_coords)
        return flare_timeareas
    
    def generate_mature_spots(self,coverage:float):
        new_spots = self.spot_generator.generate_mature_spots(coverage,self.radius)
        self.spots.add_spot(new_spots)
        self.map = self.get_pixelmap()




class SpotGenerator:
    """Spot Generator
    Class controling the birth rates and properties of new spots.
    This class is based on various studies, but since in general starspots cannot
    be resolved, lots of gaps are filled in with studies of sunspots.

    Args:
        average_area (astropy.units.quantity.Quantity [area]): average max spot area.
        area_spread (astropy.units.quantity.Quantity [area]): spread in max spot area.
        coverage (float): fractional coverage of surface by spots.
    
    Returns:
        None
    """
    def __init__(self,
    average_area:Quantity[MSH],area_spread:float,umbra_teff:Quantity[u.K],penumbra_teff:Quantity[u.K],
    growth_rate:Quantity[1/u.day]=0.52/u.day,decay_rate:Quantity[MSH/u.day]= 10.89 * MSH/u.day,
    starting_size:Quantity[MSH]=10*MSH,distribution='solar',
    coverage:float=0.2,Nlat:int=500,Nlon:int=1000,gridmaker=None
    ):
        self.average_spot_area = average_area
        self.spot_area_spread = area_spread
        self.umbra_teff = umbra_teff
        self.penumbra_teff = penumbra_teff
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.starting_size = starting_size
        self.distribution = distribution
        self.average_spot_lifetime = 2*(self.average_spot_area / self.decay_rate).to(u.hr)
        self.coverage = coverage
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat,Nlon)
        else:
            self.gridmaker = gridmaker

    def generate_spots(self,N:int)->tuple[StarSpot]:
        """ generate spots
        
        Create a specified number of StarSpot objects

        Args:
            N (int): Number of spots to create

        Returns:
            (tuple): tuple of new spots
        """
        new_max_areas = np.random.lognormal(mean=np.log(self.average_spot_area/MSH),sigma=self.spot_area_spread,size=N)*MSH
        new_r_A = np.random.normal(loc=5,scale=1,size=N)
        while np.any(new_r_A <= 0):
            new_r_A = np.random.normal(loc=5,scale=1,size=N)
        # now assign lat and lon (dist approx from 2017ApJ...851...70M)
        if self.distribution == 'solar':
            hemi = np.random.choice([-1,1],size = N)
            lat = np.random.normal(15,5,size=N)*hemi*u.deg
            lon = np.random.random(size=N)*360*u.deg
        elif self.distribution == 'iso':
            lon = np.random.random(size=N)*360*u.deg
            lats = np.arange(90)
            w = np.cos(lats*u.deg)
            lat = (np.random.choice(lats,p=w/w.sum(),size=N) + np.random.random(size=N))*u.deg * np.random.choice([1,-1],size=N)
        else:
            raise ValueError(f'Unknown value {self.distribution} for distribution')
        
        penumbra_teff = self.penumbra_teff
        umbra_teff = self.umbra_teff
        
        spots = []
        for i in range(N):
            spots.append(StarSpot(
                lat[i],lon[i],new_max_areas[i],self.starting_size,umbra_teff,penumbra_teff,
                growth_rate=self.growth_rate,decay_rate=self.decay_rate,
                r_A = new_r_A[i],Nlat=self.gridmaker.Nlat,Nlon=self.gridmaker.Nlon,gridmaker=self.gridmaker
                ))
        return tuple(spots)


    def birth_spots(self,time:Quantity[u.day],rad_star:Quantity[u.R_sun],)->tuple[StarSpot]:
        """birth spots
        Generate new StarSpot objects to be birthed in a given time.

        Args:
            time (astropy.units.quantity.Quantity [time]): amount of time in which to birth spots.
                The total number of new spots will consider this time and the birthrate
            rad_star (astropy.units.quantity.Quantity [length]): radius of star
            size_sigma (float): parameter controlling spot size distribution
            starting_size: starting size for each spot. This defaults to 10 MSH
            distribution (str): keyword controling the method for placing spots on the sphere
        
        Returns:
            (tuple): tuple of new spots
        """            
        N_exp = (self.coverage * 4*np.pi*rad_star**2 / self.average_spot_area
                        * time/self.average_spot_lifetime).to(u.Unit(''))
        # N_exp is the expectation value of N, but this is a poisson process
        N = max(0,round(np.random.normal(loc=N_exp,scale = np.sqrt(N_exp))))

        return self.generate_spots(N)        
    

    def generate_mature_spots(self,coverage:float,R_star:Quantity[u.R_sun])->List[StarSpot]:
        spots = []
        current_omega = 0*(u.deg**2)
        target_omega = (4*np.pi*coverage*u.steradian).to(u.deg**2)
        while current_omega < target_omega:
            new_spot = self.generate_spots(1)[0]
            const_spot = (new_spot.decay_rate == 0*MSH/u.day) or (new_spot.growth_rate == 0/u.day)
            if const_spot:
                area0 = self.starting_size
                area_range = new_spot.area_max - area0
                area = np.random.random()*area_range + area0
                new_spot.area_current = area
            else:
                decay_lifetime = (new_spot.area_max/new_spot.decay_rate).to(u.day)
                tau = new_spot.growth_rate
                grow_lifetime = (np.log(to_float(new_spot.area_max/self.starting_size,u.Unit('')))/tau).to(u.day)
                lifetime = grow_lifetime+decay_lifetime
                age = np.random.random() * lifetime
                new_spot.age(age)
            spots.append(new_spot)
            spot_solid_angle = new_spot.angular_radius(R_star)**2 * np.pi
            current_omega += spot_solid_angle
        return spots



class Facula:
    """facula

    Class containing model parameters of stellar faculae using the 'hot wall' model
    
    Args:
        lat (astropy.units.quantity.Quantity [angle]): latitude of facula center
        lon (astropy.units.quantity.Quantity [angle]): longitude of facula center
        Rmax (astropy.units.quantity.Quantity [length]): maximum radius of facula
        R0 (astropy.units.quantity.Quantity [length]): current radius of facula
        Zw (astropy.units.quantity.Quantity [length]): depth of the depression
        Teff_floor (astropy.units.quantity.Quantity [temperature]): effective temperature of the 'cool floor'
        Teff_wall (astropy.units.quantity.Quantity [temperature]): effective temperature of the 'hot wall'
        lifetime (astropy.units.quantity.Quantity [time]): facula lifetime
        growing (bool): whether or not the facula is still growing
        floor_threshold (astropy.units.quantity.Quantity [length]): facula radius under which the floor is no longer visible
        Nlat (int): number of latitude points. Default 500
        Nlon (int): number of longitude points Default 1000
        gridmake (CoordinateGrid): grid maker object. Default None
        
    Returns:
        None
    """
    def __init__(self,
    lat:Quantity[u.deg],lon:Quantity[u.deg],Rmax:Quantity[u.km],R0:Quantity[u.km],
    Teff_floor:Quantity[u.K],Teff_wall:Quantity[u.K],lifetime:Quantity[u.day],
    growing:bool=True,floor_threshold:Quantity[u.km] = 20*u.km,Zw:Quantity[u.km]=100*u.km,
    Nlat:int=500,Nlon:int=1000,gridmaker=None
    ):
        self.lat = lat
        self.lon = lon
        self.Rmax = Rmax
        self.current_R = R0
        self.Zw = Zw
        self.Teff_floor = self.round_teff(Teff_floor)
        self.Teff_wall = self.round_teff(Teff_wall)
        self.lifetime = lifetime
        self.is_growing = growing
        self.floor_threshold = floor_threshold
        
        if not gridmaker:
            self.gridmaker = CoordinateGrid(Nlat,Nlon)
        else:
            self.gridmaker = gridmaker
        
        latgrid,longrid = self.gridmaker.grid()
        self.r = 2* np.arcsin(np.sqrt(np.sin(0.5*(lat-latgrid))**2
                         + np.cos(latgrid)*np.cos(lat)*np.sin(0.5*(lon - longrid))**2))
        
    
    def age(self,time:Quantity[u.day]):
        """age

        progress the development of the facula by an amount of time
        
        Args:
            time (astropy.units.quantity.Quantity [time]): amount of time to progress facula
        
        Returns:
            None
        """
        if self.is_growing:
            T_from_max = -1*np.log(self.current_R/self.Rmax)*self.lifetime*0.5
            if T_from_max <= time:
                self.is_growing = False
                time = time - T_from_max
                self.current_R = self.Rmax * np.exp(-2*time/self.lifetime)
            else:
                self.current_R = self.current_R * np.exp(2*time/self.lifetime)
        else:
            self.current_R = self.current_R * np.exp(-2*time/self.lifetime)
    
    def round_teff(self,teff):
        """round teff
        Round the effective temperature for better storage
        
        Args:
            teff (astropy.units.quantity.Quantity [temperature]): temperature to round
        
        Returns:
            (astropy.units.quantity.Quantity [temperature]): rounded temperature
        """
        val = teff.value
        unit = teff.unit
        return int(round(val)) * unit
    
    def effective_area(self,angle,N=101):
        """effective area
        Calculate the effective area of the floor and walls when projected on a disk
        
        Args:
            angle (astropy.units.quantity.Quantity [angle]): angle from disk center
            N (int): number of points to sample the facula with
        
        Returns:
            (dict): effective area of the wall and floor. Keys are Teff, values are [km]**2
        """
        if self.current_R < self.floor_threshold:
            return {self.round_teff(self.Teff_floor):0.0 * u.km**2,self.round_teff(self.Teff_wall):np.pi*self.current_R**2 * np.cos(angle)}
        else:
            x = np.linspace(0,1,N) * self.current_R #distance from center along azmuth of disk
            h = np.sqrt(self.current_R**2 - x**2) # effective radius of the 1D facula approximation
            critical_angles = np.arctan(2*h/self.Zw)
            Zeffs = np.sin(angle)*np.ones(N) * self.Zw
            Reffs = np.cos(angle)*h*2 - self.Zw * np.sin(angle)
            no_floor = critical_angles < angle
            Zeffs[no_floor] = h[no_floor]*np.cos(angle)
            Reffs[no_floor] = 0
            
            return {self.round_teff(self.Teff_wall): np.trapz(Zeffs,x),self.round_teff(self.Teff_floor): np.trapz(Reffs,x)}
    
    def fractional_effective_area(self,angle,N=101):
        """fractional effective area
        effective area as a fraction of the projected area of a region of quiet photosphere with the same radius
        and distance from limb
        
        Args:
            angle (astropy.units.quantity.Quantity [angle]): angle from disk center
            
        Returns:
            (dict): fractional effective area of the wall and floor. Keys are Teff
        """
        effective_area = self.effective_area(angle,N=N)
        frac_eff_area = {}
        total = 0
        for teff in effective_area.keys():
            total = total + effective_area[teff]
        for teff in effective_area.keys():
            frac_eff_area[teff] = (effective_area[teff]/total).to(u.Unit(''))
        return frac_eff_area
    def angular_radius(self,star_rad):
        """angular radius
        Calculate the anglular radius
        
        Args:
            star_rad (astropy.units.qunatity.Quantity [length]): radius of the star
        
        Returns
            (astropy.units.qunatity.Quantity [angle]): angular radius of the facula
        """
        return self.current_R/star_rad * 180/np.pi * u.deg
    def map_pixels(self,star_rad):
        """map pixels
        
        """
#         latgrid,longrid = self.gridmaker.grid()
#         lat0 = self.lat
#         lon0 = self.lon
        rad = self.angular_radius(star_rad)
        
#         a = np.cos(lat0 - latgrid)
#         b = np.cos(lat0 + latgrid)
#         c = np.cos(lon0 - longrid)
#         r = 2*np.arcsin(np.sqrt(0.5*(1+(1+0.5+0.5*c)*a) + 0.5*(1+c)*b))
        
#         r = 2* np.arcsin(np.sqrt(np.sin(0.5*(lat0-latgrid))**2
#                          + np.cos(latgrid)*np.cos(lat0)*np.sin(0.5*(lon0 - longrid))**2))
        pix_in_fac = self.r<=rad
        return pix_in_fac

class FaculaCollection:
    """Facula Collection
    Containter class to store faculae
    
    Args:
        *faculae (tuple): series of faculae objects
        Nlat (int): number of latitude points. Default 500
        Nlon (int): number of longitude points Default 1000
        gridmake (CoordinateGrid): grid maker object. Default None
    
    Returns:
        None
    """
    def __init__(self,*faculae,Nlat=500,Nlon=1000,gridmaker=None):
        self.faculae = tuple(faculae)
        
        if not gridmaker:
            self.gridmaker = CoordinateGrid(Nlat,Nlon)
        else:
            self.gridmaker = gridmaker
        for facula in faculae:
            facula.gridmaker = self.gridmaker
    def add_faculae(self,facula):
        """add facula(e)
        Add a facula or faculae

        Args:
            facula (Facula or series of Facula): Facula object(s) to add
        
        Returns:
            None
        """
        if isinstance(facula, Facula):
            facula.gridmaker = self.gridmaker
        else:
            for fac in facula:
                fac.gridmaker = self.gridmaker
        self.faculae += tuple(facula)
    def clean_faclist(self):
        """clean faculae list
        Remove faculae that have decayed to Rmax/e**2 radius.

        Args:
            None
        
        Returns:
            None
        """
        faculae_to_keep = []
        for facula in self.faculae:
            if (facula.current_R <= facula.Rmax/np.e**2) and (not facula.is_growing):
                pass
            else:
                faculae_to_keep.append(facula)
        self.faculae = faculae_to_keep
    def age(self,time):
        """age
        Age spots according to its growth timescale and decay rate.
        Remove spots that have decayed.

        Args:
            time (astropy.units.quantity.Quantity [time]): length of time to age the spot.
                For most realistic behavior, time should be << spot lifetime
        
        Returns:
            None
        """
        for facula in self.faculae:
            facula.age(time)
        self.clean_faclist()
    def map_pixels(self,pixmap,star_rad,star_teff):
        """map_pixels
        Map facula parameters to pixel locations
        
        Args:
            latgrid (astropy.units.quantity.Quantity [angle], shape(M,N)): grid of latitude points
            longrid (astropy.units.quantity.Quantity [angle], shape(M,N)): grid of longitude points
            pixmap (astropy.units.quantity.Quantity [temperature], shape(M,N)): grid of effctive temperature
            star_rad (astropy.units.quantity.Quantity [length]): radius of the star
            star_teff (astropy.units.quantity.Quantity [temperature]): temperature of quiet stellar photosphere
        
        Returns:
            (np.ndarray [int8], shape(M,N)): grid of integer keys showing facula loactions
            (dict): dictionary maping index in self.faculae to the integer grid of facula locations
        """
        int_map = self.gridmaker.zeros(dtype='int16')
        map_dict = {}
        for i in range(len(self.faculae)):
            facula = self.faculae[i]
            pix_in_fac = facula.map_pixels(star_rad)
            is_photosphere = pixmap == star_teff
            int_map[pix_in_fac & is_photosphere] = i+1
            map_dict[i] = i+1
        return int_map, map_dict

class FaculaGenerator:
    """ Facula generator

    Class controling the birth rates and properties of new faculae.
    Radius distribution from K. P. Topka et al 1997 ApJ 484 479
    Lifetime distribution from 2022SoPh..297...48H
    
    Args:
        R_peak (astropy.unit.quantity.Quantity [length]): Radius to use as the peak of the distribution
        R_HWHM (astropy.unit.quantity.Quantity [length]): Radius half width half maximum. Difference between
            the peak of the radius distribution and the half maximum in the positive direction
        T_peak (astropy.unit.quantity.Quantity [time]): Lifetime to use as the peak of the distribution
        T_HWHM (astropy.unit.quantity.Quantity [time]): Lifetime half width half maximum. Difference between
            the peak of the lifetime distribution and the half maximum in the positive direction
        coverage (float): fraction of the stellar surface covered by faculae
        dist (str): type of distribution
        
    """
    def __init__(self,R_peak:Quantity[u.km] = 800*u.km, R_HWHM:Quantity[u.km] = 300*u.km,
                 T_peak:Quantity[u.hr] = 6.2*u.hr, T_HWHM:Quantity[u.hr] = 4.7*u.hr,
                 coverage:float=0.0001,dist:str = 'iso',Nlon:int=1000,Nlat:int=500,gridmaker=None,
                 teff_bounds = (2500*u.K,3400*u.K)):
        assert u.get_physical_type(R_peak) == 'length'
        assert u.get_physical_type(R_HWHM) == 'length'
        assert u.get_physical_type(T_peak) == 'time'
        assert u.get_physical_type(T_HWHM) == 'time'
        self.radius_unit = u.km
        self.lifetime_unit = u.hr
        self.R0 = np.log10(R_peak/self.radius_unit)
        self.sig_R = np.log10((R_peak + R_HWHM)/self.radius_unit) - self.R0
        self.T0 = np.log10(T_peak/self.lifetime_unit)
        self.sig_T = np.log10((T_peak + T_HWHM)/self.lifetime_unit) - self.T0
        assert isinstance(coverage,float)
        self.coverage = coverage
        self.dist = dist
        if gridmaker is None:
            self.gridmaker = CoordinateGrid(Nlat,Nlon)
        else:
            self.gridmaker=gridmaker
        self.Nlon = Nlon
        self.Nlat = Nlat
        self.teff_bounds = teff_bounds
        
    def get_floor_teff(self,R,Teff_star):
        """Get floor Teff
        Get the Teff of the faculae floor based on the radius and photosphere Teff
        Based on K. P. Topka et al 1997 ApJ 484 479
        
        Args:
            R (astropy.unit.quantity.Quantity [length]): radius of the facula[e]
            Teff_star (astropy.unit.quantity.Quantity [temperature]): effective temperature of the photosphere
        
        Returns:
            (astropy.unit.quantity.Quantity [temperature]): floor temperature of faculae
        """
        d_teff = np.zeros(len(R)) * u.K
        reg = R <= 150*u.km
        d_teff[reg] = -1 * u.K * R[reg]/u.km/5
        reg = (R > 150*u.km) & (R <= 175*u.km )
        d_teff[reg] = 510 * u.K - 18*R[reg]/5/u.km*u.K
        reg = (R > 175*u.km)
        d_teff[reg] = -4*u.K*R[reg]/7/u.km - 20 * u.K
        teff = d_teff + Teff_star
        teff = np.clip(teff,*self.teff_bounds)
        return teff
    
    def get_wall_teff(self,R,Teff_floor):
        """Get wall Teff
        Get the Teff of the faculae wall based on the radius and floor Teff
        Based on K. P. Topka et al 1997 ApJ 484 479
        
        Args:
            R (astropy.unit.quantity.Quantity [length]): radius of the facula[e]
            Teff_floor (astropy.unit.quantity.Quantity [temperature]): effective temperature of the cool floor
        
        Returns:
            (astropy.unit.quantity.Quantity [temperature]): wall temperature of faculae
        """
        teff = Teff_floor + R/u.km * u.K + 125*u.K
        teff = np.clip(teff,*self.teff_bounds)
        return teff
        
        
    def birth_faculae(self,time, rad_star, Teff_star):
        """birth faculae
        determine the number and parameters of faculae to create in an amount of time
        
        Args:
            time (astropy.unit.quantity.Quantity [time]): time over which to create faculae
            rad_star (astropy.unit.quantity.Quantity [length]): radius of the star
            Teff_star (astropy.unit.quantity.Quantity [temperature]): temperature of the star
        
        Returns:
            (tuple): tuple of new faculae
        """
        N_exp = (self.coverage * 4*np.pi*rad_star**2 / ((10**self.R0*self.radius_unit)**2 * np.pi)
                        * time/(10**self.T0 * self.lifetime_unit * 2)).to(u.Unit(''))
#         print((self.coverage * 4*np.pi*rad_star**2 / ((10**self.R0*self.radius_unit)**2 * np.pi)).to(u.Unit('')))
#         print((time/(10**self.T0 * self.lifetime_unit)).to(u.Unit('')))
        # N_exp is the expectation value of N, but this is a poisson process
        N = max(0,round(np.random.normal(loc=N_exp,scale = np.sqrt(N_exp))))
#         print(f'{N_exp:.2f}-->{N}')
        mu = np.random.normal(loc=0,scale=1,size=N)
        max_radii = 10**(self.R0 + self.sig_R * mu) * self.radius_unit
        lifetimes = 10**(self.T0 + self.sig_T * mu) * self.lifetime_unit
        starting_radii = max_radii / np.e**2
        lats = None
        lons = None
        if self.dist == 'iso':
            x = np.linspace(-90,90,180,endpoint=False)*u.deg
            p = np.cos(x)
            lats = (np.random.choice(x,p=p/p.sum(),size=N) + np.random.random(size=N)) * u.deg
            lons = np.random.random(size=N) * 360 * u.deg
        else:
            raise NotImplementedError(f'{self.dist} has not been implemented as a distribution')
        teff_floor = self.get_floor_teff(max_radii,Teff_star)
        teff_wall = self.get_wall_teff(max_radii,teff_floor)
        new_faculae = []
        for i in range(N):
            new_faculae.append(Facula(lats[i], lons[i], max_radii[i], starting_radii[i], teff_floor[i],
                                     teff_wall[i],lifetimes[i], growing=True, floor_threshold=20*u.km, Zw = 100*u.km,
                                     Nlon=self.Nlon,Nlat=self.Nlat))
        return tuple(new_faculae)
        
class CoordinateGrid:
    """Coordinate Grid
    Class to standardize the creation of latitude and longitude grids
    
    Args:
        Nlat (int): number of latitude points
        Nlon (int): number of longitude points
    
    Returns:
        None
    """
    def __init__(self,Nlat=500,Nlon=1000):
        assert isinstance(Nlat,int)
        assert isinstance(Nlon,int)
        self.Nlat = Nlat
        self.Nlon = Nlon
    
    def oned(self):
        """1-D
        Create one dimensional arrays of latitude and longitude points
        
        Args:
            None
        
        Returns:
            (astropy.unit.quantity.Quantity [angle], shape=(Nlat)): array of latitude points
            (astropy.unit.quantity.Quantity [angle], shape=(Nlon)): array of longitude points
        """
        lats = np.linspace(-90,90,self.Nlat)*u.deg
        lons = np.linspace(0,360,self.Nlon)*u.deg
        return lats,lons
    
    def grid(self):
        """grid
        Create a 2 dimensional grid of latitude and longitude points:
        
        Args:
            None
        
        Returns:
            astropy.unit.quantity.Quantity [angle], shape=(Nlat)): array of latitude points
            astropy.unit.quantity.Quantity [angle], shape=(Nlat)): array of longitude points
        """
        lats, lons = self.oned()
        return np.meshgrid(lats,lons)
    def zeros(self,dtype='float32'):
        """zeros
        Get a grid of zeros
        
        Args:
            dtype (str): data type to pass to np.zeros
        
        Returns:
            (np.ndarray, shape=(self.Nlat,self.Nlon)): grid of zeros
        """
        return np.zeros(shape=(self.Nlon,self.Nlat),dtype=dtype)
    def __eq__(self,other):
        """equals
        Check to see if two CoordinateGrid objects are equal
        
        Args:
            other (CoordinateGrid): another CoordinateGrid object
        
        Returns:
            (bool): whether the two objects have equal properties
        """
        if not isinstance(other,CoordinateGrid):
            return False
        else:
            return (self.Nlat == other.Nlat) & (self.Nlon == other.Nlon)

class StellarFlare:
    """ Class to store and control stellar flare information

        Args:
            fwhm (Quantity): Full-width-half-maximum of the flare
            energy (Quantity): time-integrated bolometric energy
            lat (Quantity): Latitude of flare on star
            lon (Quantity): Longitude of flare on star
            Teff (Quantity): Blackbody temperature
            tpeak (Quantity): time of the flare peak
    
    """
    def __init__(self,fwhm:Quantity,energy:Quantity,lat:Quantity,lon:Quantity,Teff:Quantity,tpeak:Quantity):
        self.fwhm = fwhm
        self.energy = energy
        self.lat = lat
        self.lon = lon
        self.Teff = Teff
        self.tpeak = tpeak
    
    def calc_peak_area(self)->u.Quantity[u.km**2]:
        """
        Calc peak area

        Calculate the flare area at its peak

        Args:
            None
        
        Returns:
            (Quantity): peak flare area
        """
        time_area = self.energy/c.sigma_sb/(self.Teff**4)
        area_std = 1*u.km**2
        time_area_std = flareintegral(self.fwhm,area_std)
        area = time_area/time_area_std * area_std
        return area.to(u.km**2)
    
    def areacurve(self,time:Quantity[u.hr]):
        """
        areacurve

        Compute the flare area as a function of time

        Args:
            time (Quantity): times at which to sample the area
        
        Returns
            (Qunatity): Area at each time
        """
        t_unit = u.day # this is the unit of xoflares
        a_unit = u.km**2
        peak_area = self.calc_peak_area()
        areas = get_light_curvenp(to_float(time,t_unit),
                                        [to_float(self.tpeak,t_unit)],
                                        [to_float(self.fwhm,t_unit)],
                                        [to_float(peak_area,a_unit)])
        return areas * a_unit
    
    def get_timearea(self,time:Quantity[u.hr]):
        """
        Get timearea

        Calcualte the integrated time*area of the flare

        Args:
            time (Quantity): times at which to sample the area
        
        Returns
            (Qunatity): integrated timearea
        """
        areas = self.areacurve(time)
        timearea = np.trapz(areas,time)
        return timearea.to(u.Unit('hr km2'))
    
class FlareGenerator:
    """ Class to decide when a flare occurs and its magnitude

    Args:
        stellar_teff (Quantity): Temperature of the star
        stellar_rot_period (Quantity): Rotation period of the star
        prob_following (float): probability of a flare being closely followed by another flare
        mean_teff (Quantity): Mean teff of the set of flares
        sigma_teff (Quantity): Standard deviation of the flare teffs
        mean_log_fwhm_days (float): mean of the log(fwhm/day) distribution
        sigma_log_fwhm_days (float): standard deviation of the log(fwhm/day) distribution
        log_E_erg_max (float): Maximum log(E/erg) to draw from
        log_E_erg_min (float): Minimum log(E/erg) to draw from
        log_E_erg_Nsteps (float): Number of samples in the log(E/erg) array. 0 disables flares.

    """
    def __init__(self,stellar_teff:Quantity,stellar_rot_period:Quantity, prob_following = 0.5,
                mean_teff = 9000*u.K, sigma_teff = 500*u.K,mean_log_fwhm_days=-0.85,sigma_log_fwhm_days=0.3,
                log_E_erg_max=36, log_E_erg_min = 33, log_E_erg_Nsteps=100):
        """ FWHM data from Table 2 of Gunther et al. 2020, AJ 159 60
        """
        self.stellar_teff = stellar_teff
        self.stellar_rot_period = stellar_rot_period
        self.prob_following = prob_following
        self.mean_teff = mean_teff
        self.sigma_teff = sigma_teff
        self.mean_log_fwhm_days = mean_log_fwhm_days
        self.sigma_log_fwhm_days = sigma_log_fwhm_days
        self.log_E_erg_max = log_E_erg_max
        self.log_E_erg_min = log_E_erg_min
        self.log_E_erg_Nsteps = log_E_erg_Nsteps
    def powerlaw(self, E:Quantity):
        """
        powerlaw

        Generate a flare frequency distribution.
        Based on Gao+2022 TESS corrected data

        Args:
            E (Quantity): Array of Energy values
        
        Returns:
            (Quantity): Matching array of frequencies
        """
        alpha = -0.829
        beta = 26.87
        logfreq = beta + alpha*np.log10(E/u.erg)
        freq = 10**logfreq / u.day
        return freq
    def get_flare(self,Es:Quantity,time:Quantity):
        """
        get flare

        Generate a flare in some time duration

        Args:
            Es (Quantity): Energies to draw from
            time (Quantity): time duration
        
        Returns:
            (Quantity): Energy of a generated flare
        """
        Nexp = to_float(self.powerlaw(Es) * time,u.Unit(''))
        # N_previous = 1
        E_final = 0*u.erg
        for e, N in zip(Es,Nexp):
            if np.round(np.random.normal(loc=N,scale=np.sqrt(N))) > 0:
                E_final = e
            else:
                break
            # if np.random.random() < N/N_previous:
            #     # f_previous = f
            #     E_final = e
            # else:
            #     break
        return E_final
    def generate_flares(self,Es:Quantity,time:Quantity):
        """ 
        generate flares

        Generate a group of flare(s)
        valid if flare length is much less than time

        Args:
            Es (Quantity): Energies to draw from
            time (Quantity): time duration
        
        Returns:
            (Quantity): Energies of generated flares
        """
        unit=u.erg
        flare_energies = []
        E = self.get_flare(Es,time)
        if E == 0*u.erg:
            return flare_energies
        else:
            flare_energies.append(to_float(E,unit))
            cont = np.random.random() < self.prob_following
            while cont:
                while True:
                    E = self.get_flare(Es,time)
                    if E == 0*u.erg:
                        pass
                    else:
                        flare_energies.append(to_float(E,unit))
                        cont = np.random.random() < self.prob_following
                        break
            return flare_energies*unit
    
    def generate_coords(self):
        """
        generate coords

        Generate random coordinates for the flare

        Args:
            None
        
        Returns:
            (Quantity): latitude
            (Quantity): longitude
        """
        lon = np.random.random()*360*u.deg
        lats = np.arange(90)
        w = np.cos(lats*u.deg)
        lat = (np.random.choice(lats,p=w/w.sum()) + np.random.random())*u.deg * np.random.choice([1,-1])
        return lat,lon
    
    def generate_fwhm(self):
        """
        generate fwhm
        
        Pull a FWHM value from a distribution

        Args:
            None

        Returns:
            (Quantity): FWHM
        """
        log_fwhm_days = np.random.normal(loc=self.mean_log_fwhm_days,scale=self.sigma_log_fwhm_days)
        fwhm = 10**log_fwhm_days * u.day
        return fwhm
    
    def generate_flare_set_spacing(self):
        """
        generate flare set spacing

        Isolated flares are random events, but if you see one flare, it is likely you will see another
            soon after. How soon? This distribution will tell you.
            
            The hope is that as we learn more this will be set by the user
        
        Args:
            None
        
        Returns:
            (Quantity): time between flares
        """
        while True: # this cannot be a negative value. We will loop until we get something positive (usually unneccessary)
            spacing = np.random.normal(loc=4,scale=2)*u.hr
            if spacing > 0*u.hr:
                return spacing
    def generage_E_dist(self):
        """
        generate Energy distribution

        Generate the energy distribution based on parameters

        Args:
            None
        
        Returns:
            (Quantity): Series of energies
        """
        return np.logspace(self.log_E_erg_min - 0.2,self.log_E_erg_max,self.log_E_erg_Nsteps)*u.erg

    def generate_teff(self):
        """ 
        generate teff

        Randomly generate teff, round to int

        Args:
            None
        
        Returns:
            (Quantity): teff
        """
        assert self.mean_teff > 0*u.K # prevent looping forever if user gives bad parameters
        while True: # this cannot be a negative value. We will loop until we get something positive (usually unneccessary)
            teff = np.random.normal(loc=to_float(self.mean_teff,u.K),scale=to_float(self.sigma_teff,u.K))
            teff = int(np.round(teff)) * u.K
            if teff > 0*u.K:
                return teff

    def generate_flare_series(self,Es:Quantity,time:Quantity):
        """
        generate flare series

        Generate as many flares within a duration of time as can be fit given computed frequencies

        Args:
            Es (Quantity): Series of energies
            time (Quantity): Time series to create flares in

        Returns:
            (list of Flares): list of created stellar flares
        """
        flares = []
        tmin=0*u.s
        tmax = time
        timesets = [[tmin,tmax]]
        while True:
            next_timesets = []
            N  = len(timesets)
            for i in range(N): # loop thought blocks of time
                timeset = timesets[i]
                dtime = timeset[1] - timeset[0]
                flare_energies = self.generate_flares(Es,dtime)
                if len(flare_energies) > 0:
                    base_tpeak = np.random.random()*dtime + timeset[0]
                    peaks = [deepcopy(base_tpeak)]
                    for j in range(len(flare_energies)): #loop through flares
                        if j > 0:
                            base_tpeak = base_tpeak + self.generate_flare_set_spacing()
                            peaks.append(deepcopy(base_tpeak))
                        energy = flare_energies[j]
                        lat,lon = self.generate_coords()
                        fwhm = self.generate_fwhm()
                        teff = self.generate_teff()
                        if np.log10(to_float(energy,u.erg)) >= self.log_E_erg_min:
                            flares.append(StellarFlare(fwhm,energy,lat,lon,teff,base_tpeak))
                    next_timesets.append([timeset[0],min(peaks)])
                    if max(peaks) < timeset[1]:
                        next_timesets.append([max(peaks),timeset[1]])
                else:
                    pass # there are no flares during this time
            timesets = next_timesets
            if len(timesets) == 0:
                return flares
        

class FlareCollection:
    """ This class stores a series of flares and does math to turn them into lightcurves

    Args:
        flares (list of StellarFlare or StellarFlare): flares
    """
    def __init__(self,flares:Typing.Union[List[StellarFlare],StellarFlare]):
        if isinstance(flares,StellarFlare):
            self.flares = [flares]
        else:
            self.flares=flares
        self.index()

    
    def index(self):
        """
        index

        Get peak times and fwhms for flares as arrays

        Args:
            None

        Returns:
            None
        """
        tpeak = []
        fwhm = []
        unit = u.hr
        for flare in self.flares:
            tpeak.append(to_float(flare.tpeak,unit))
            fwhm.append(to_float(flare.fwhm,unit))
        tpeak = np.array(tpeak)*unit
        fwhm = np.array(fwhm)*unit
        self.peaks = tpeak
        self.fwhms = fwhm
    
    def mask(self, tstart: Quantity[u.hr], tfinish: Quantity[u.hr]):
        """
        mask

        Create a boolean mask to indicate which flares are visible within a certain time period

        Args:
            tstart (Quantity): Starting time
            tfinish (Quantity): Ending time
        
        Returns:
            (np.array): boolean array of visible flares
        """
        padding_after = 10 # number of fwhm ouside this range a flare peak can be to still be included
        padding_before = 20
        after_start = self.peaks + padding_before*self.fwhms > tstart
        before_end = self.peaks - padding_after*self.fwhms < tfinish
        
        return after_start & before_end
    
    def get_flares_in_timeperiod(self,tstart: Quantity[u.hr], tfinish: Quantity[u.hr])-> List[StellarFlare]:
        """
        get flares in timeperiod

        Generate a mask and select flares without casting to np.array (list comp instead)

        Args:
            tstart (Quantity): Starting time
            tfinish (Quantity): Ending time
        
        Returns:
            (list of StellarFlare): flares that occur
        """
        mask = self.mask(tstart,tfinish)
        masked_flares = [flare for flare, include in zip(self.flares,mask) if include]
        # essentially the same as self.flares[mask], but without casting to ndarray
        return masked_flares
    
    def get_visible_flares_in_timeperiod(self,tstart: Quantity[u.hr], tfinish: Quantity[u.hr],
                                        sub_obs_coords={'lat':0*u.deg,'lon':0*u.deg})-> List[StellarFlare]:
        """
        get visible flares in timeperiod

        Select flares that occur in a certain timeperiod on a given hemisphere

        Args:
            tstart (Quantity): Starting time
            tfinish (Quantity): Ending time
            sub_obs_coords(dict): coordinates defining the hemisphere
        
        Returns:
            (list of StellarFlare): flares that occur and are visible by observer
        """
        masked_flares = self.get_flares_in_timeperiod(tstart,tfinish)
        visible_flares = []
        for flare in masked_flares:
            cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(flare.lat)
                + np.cos(sub_obs_coords['lat'])* np.cos(flare.lat)
                 * np.cos(sub_obs_coords['lon']-flare.lon) )
            if cos_c > 0: # proxy for angular radius that has low computation time
                visible_flares.append(flare)
        return visible_flares
    
    def get_flare_integral_in_timeperiod(self,tstart: Quantity[u.hr], tfinish: Quantity[u.hr],
                                        sub_obs_coords={'lat':0*u.deg,'lon':0*u.deg}):
        """
        get flare integral in timeperiod

        Calculate the integrated time-area of flares in a timeperiod

        Args:
            tstart (Quantity): Starting time
            tfinish (Quantity): Ending time
            sub_obs_coords(dict): coordinates defining the hemisphere
        
        Returns:
            (Quantity): integrated time-area
        """
        visible_flares = self.get_visible_flares_in_timeperiod(tstart,tfinish,sub_obs_coords)
        flare_timeareas = []
        time_resolution = 10*u.min
        N_steps = int(((tfinish-tstart)/time_resolution).to(u.Unit('')).value)
        time = np.linspace(tstart,tfinish,N_steps)
        for flare in visible_flares:
            timearea = flare.get_timearea(time)
            flare_timeareas.append(dict(Teff=flare.Teff,timearea=timearea))
        return flare_timeareas

    




            
                        






        

