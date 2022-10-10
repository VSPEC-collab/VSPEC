import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u, constants as c
from astropy.units.quantity import Quantity as quant
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
        T (astropy.units.quantity.Quantity [time]): Spt decay timescale (deprecated).
        r_A (float): Ratio of total spot area to umbra area. From 2013PhDT.......359G = 5+/-1 (compiled from various sources)
        growing (bool): Whether or not the spot is growing.
        growth_rate (astropy.units.quantity.Quantity [1/time]): Fractional growth of the spot for a given unit time.
            From 2013PhDT.......359G can be 0.52/day to 1.83/day (compiled form various sources)

    Returns:
        None
    """
    def __init__(self,lat,lon,Amax,A0,Teff_umbra,Teff_penumbra,T = 1*u.day,r_A=5,growing=True,growth_rate = 0.52/u.day):
        assert isinstance(lat,quant)
        assert isinstance(lon,quant)
        self.coords = {'lat':lat,'lon':lon}
        assert isinstance(Amax,quant)
        assert isinstance(A0,quant)
        self.area_max = Amax
        self.area_current = A0
        assert isinstance(Teff_umbra,quant)
        assert isinstance(Teff_penumbra,quant)
        self.Teff_umbra = Teff_umbra
        self.Teff_penumbra = Teff_penumbra
        assert isinstance(T,quant)
        self.decay_timescale = T
        self.decay_rate = 10.89 * MSH/u.day
        self.total_area_over_umbra_area = r_A
        self.is_growing = growing
        self.growth_rate = growth_rate
    def radius(self):
        """radius
        Get the radius of the spot.

        Args:
            None
        
        Returns:
            (astropy.units.quantity.Quantity [length]): Radius of spot.
        """
        return np.sqrt(self.area_current/np.pi).to(u.km)
    def angular_radius(self,star_rad):
        """angular radius
        Get the angular radius of the spot

        Args:
            star_rad (astropy.units.quantity.Quantity [length]): radius of the star.
        
        Returns:
            (astropy.units.quantity.Quantity [angle]): angular radius of the spot
        """
        radius = self.radius()
        angle_in_rad = radius/star_rad
        return angle_in_rad/np.pi * 180 *u.deg
    
    def map_pixels(self,latgrid,longrid,star_rad):
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
        lat0 = self.coords['lat']
        lon0 = self.coords['lon']
        r = 2* np.arcsin(np.sqrt(np.sin(0.5*(lat0-latgrid))**2
                         + np.cos(latgrid)*np.cos(lat0)*np.sin(0.5*(lon0 - longrid))**2))
        return {self.Teff_umbra:r < radius_umbra,
                self.Teff_penumbra: r < radius}
    def surface_fraction(self,sub_obs_coords,star_rad,N=1001):
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
            (floar): fraction of observed disk covered by spot
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
        return (np.trapz(integrand,x=c)/(2*np.pi*u.steradian)).to(u.Unit(''))
        
    
    def surface_fraction_old(self,sub_obs_coords,star_rad):
        """surface fraction -- old version
        Determine the surface fraction covered by a spot from a given angle of observation.
        This algorithm diverges from reality for large spots and spots on the limbs.

        Args:
            sub_obs_coord (dict): dictionary giving coordinates of the sub-observation point.
                This is the point that is at the center of the stellar disk from the view of
                an observer. Format: {'lat':lat,'lon':lon} where lat and lon are astropy Quantity objects
            star_rad (astropy.units.quantity.Quantity [length]): radius of the star.
            N (int): number of points to use in numerical integration. N=1000 is not so different from N=100000.
        
        Returns:
            (float): fraction of observed disk covered by spot
        """
        x0 = np.cos(self.coords['lat'])*np.cos(self.coords['lon'])
        x1 = np.cos(sub_obs_coords['lat'])*np.cos(sub_obs_coords['lon'])
        y0 = np.cos(self.coords['lat'])*np.sin(self.coords['lon'])
        y1 = np.cos(sub_obs_coords['lat'])*np.sin(sub_obs_coords['lon'])
        z0 = np.sin(self.coords['lat'])
        z1 = np.sin(sub_obs_coords['lat'])
        cos_alpha = x0*x1 + y0*y1 + z0*z1
        if cos_alpha < 0:
            return 0
        else:
            effective_size = self.area_current * cos_alpha
            return (effective_size/(2*np.pi*star_rad**2)).to(u.Unit(''))
    
    def age(self,time):
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
    
    Returns:
        None
    """
    def __init__(self,*spots):
        self.spots = spots
    def add_spot(self,spot):
        """add spot
        Add a spot

        Args:
            spot (StarSpot or series of StarSpot): StarSpot object(s) to add
        
        Returns:
            None
        """
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
    def map_pixels(self,latgrid,longrid,star_rad,star_teff):
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
        surface_map = np.ones(shape=latgrid.shape) * star_teff
        for spot in self.spots:
            teff_dict = spot.map_pixels(latgrid,longrid,star_rad)
            #penumbra
            assign = teff_dict[spot.Teff_penumbra] & (surface_map > spot.Teff_penumbra)
            surface_map[assign] = spot.Teff_penumbra
            #umbra
            assign = teff_dict[spot.Teff_umbra] & (surface_map > spot.Teff_umbra)
            surface_map[assign] = spot.Teff_umbra
        return surface_map
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
        resolution (dict): resolution of latitude and longitude points for pixel map. Default {'lat':500,'lon':1000}
    
    Returns:
        None
    """
    def __init__(self,Teff,radius,period,spots,name='',distance = 1*u.pc,resolution = {'lat':500,'lon':1000}):
        self.name = name
        assert isinstance(Teff,quant)
        self.Teff = Teff
        assert isinstance(radius,quant)
        self.radius = radius
        assert isinstance(distance,quant)
        self.distance = distance
        assert isinstance(period,quant)
        self.period = period
        assert isinstance(spots,SpotCollection)
        self.spots = spots
        
        self.resolution = resolution
        self.map = self.get_pixelmap(Nlat=self.resolution['lat'],Nlon=self.resolution['lon'])
        
        self.spot_generator = SpotGenerator(500*MSH,200*MSH,coverage=0.15)
        
    def get_pixelmap(self,Nlat=500,Nlon=1000):
        """get pixelmap
        Create map of stellar surface based on spots:
        
        Args:
            Nlat (int): Number of latitude points to map. Default 500
            Nlon (int): Number of longitude points to map. Default 1000
        
        Returns:
            (array of astropy.units.quantity.Quantity [temperature], Shape(Nlon,Nlat)): Map of stellar surface with Teff assigned to each pixel.

        """
        lats = np.linspace(-90,90,Nlat) * u.deg
        lons = np.linspace(0,360,Nlon)*u.deg
        latgrid,longrid = np.meshgrid(lats,lons)
        return self.spots.map_pixels(latgrid,longrid,self.radius,self.Teff)
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
        self.map = self.get_pixelmap(self.resolution['lat'],self.resolution['lon'])
    def add_spot(self,spot):
        """add spot
        Add a spot

        Args:
            spot (StarSpot or series of StarSpot): StarSpot object(s) to add
        
        Returns:
            None
        """
        assert isinstance(spot,StarSpot)
        self.spots.add_spot(spot)
        self.map = self.get_pixelmap(self.resolution['lat'],self.resolution['lon'])
    def calc_coverage(self,sub_obs_coords,Nlat=500,Nlon=1000):
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
        lats = np.linspace(-90,90,Nlat) * u.deg
        lons = np.linspace(0,360,Nlon)*u.deg
        latgrid,longrid = np.meshgrid(lats,lons)
        cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(latgrid)
                + np.cos(sub_obs_coords['lat'])* np.cos(latgrid)
                 * np.cos(sub_obs_coords['lon']-longrid) )
        cos_c[cos_c < 0] = 0
        jacobian = np.sin(latgrid + 90*u.deg)
        Teffs = np.unique(self.map)
        data = {}
        total = 0
        for teff in Teffs:
            pix = self.map == teff
            pix_sum = (pix.astype('float32') * cos_c * jacobian).sum()
            total += pix_sum
            data[teff] = pix_sum
        # normalize
        for teff in Teffs:
            data[teff] = data[teff]/total
        return data
    
    def calc_orthographic_mask(self,sub_obs_coords,Nlat=500,Nlon=1000):
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
        lats = np.linspace(-90,90,Nlat) * u.deg
        lons = np.linspace(0,360,Nlon)*u.deg
        latgrid,longrid = np.meshgrid(lats,lons)
        cos_c = (np.sin(sub_obs_coords['lat']) * np.sin(latgrid)
                + np.cos(sub_obs_coords['lat'])* np.cos(latgrid)
                 * np.cos(sub_obs_coords['lon']-longrid) )
        cos_c[cos_c < 0] = 0
        return cos_c

    def birth_spots(self,time):
        """birth spots
        Create new spots from a spot generator.

        Args:
            time (astropy.units.quantity.Quantity [time]): time over which these spots should be created.
        
        Returns:
            None
        """
        self.spots.add_spot(self.spot_generator.birth_spots(time,self.radius,self.period,self.Teff))
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

class SpotGenerator:
    """Spot Generator
    Class controling the birth rates and properties of new spots.
    This class is based on various studies, but since in general starspots cannot
    be resolved, lots of gaps are filled in with studies of sunspots.

    Args:
        average_area (astropy.units.quantity.Quantity [area]): average max spot area.
        area_spread (astropy.units.quantity.Quantity [area]): spread in max spot area.
        coverage (float): fractional coverage of surface by spots. Default None. In that
            case the coverage is calculated based on the rotation rate.
    
    Returns:
        None
    """
    def __init__(self,average_area,area_spread,coverage=None):
        self.average_spot_area = average_area
        self.spot_area_spread = area_spread
        self.decay_rate = 10.89 * MSH/u.day
        self.average_spot_lifetime = 2*(self.average_spot_area / self.decay_rate).to(u.hr)
        self.coverage = coverage
    def get_variability(self,rotation_period):
        """get variability
        Get variability from stellar rotaion period based on imperical relation
        found by Nichols-Flemming & Blackman 2020, 2020MNRAS.491.2706N
        
        Args:
            rotation_period (astropy.units.quantity.Quantity [time]): stellar rotation period.
        
        Returns:
            (float) variability as a fraction for M dwarfs
        """
        x = (rotation_period/u.day).to(u.Unit(''))**2
        y = 13.91 * x**(-0.30)
        return y/100

    def birth_spots(self,time,rad_star,rotation_period_star,Teff_star,sigma = 0.2,starting_size=10*MSH):
        """birth spots
        Generate new StarSpot objects to be birthed in a given time.

        Args:
            time (astropy.units.quantity.Quantity [time]): amount of time in which to birth spots.
                The total number of new spots will consider this time and the birthrate
            rad_star (astropy.units.quantity.Quantity [length]): radius of star
            rotation_period (astropy.units.quantity.Quantity [time]): stellar rotation period
            Tef_star (astropy.units.quantity.Quantity [temperature]): effective temperature of the star
            sigma (float): parameter controlling spot size distribution
            starting_size: starting size for each spot. This defaults to 10 MSH
        
        Returns:
            (tuple): tuple of new spots
        """    
        N_exp=0
        
        if self.coverage:
            N_exp = (self.coverage * 4*np.pi*rad_star**2 / self.average_spot_area
                        * time/self.average_spot_lifetime).to(u.Unit(''))
        else:
            N_exp = (4*np.pi*rad_star**2 * self.get_variability(rotation_period_star)/self.average_spot_area
                        * time/self.average_spot_lifetime).to(u.Unit(''))
        # N_exp is the expectation value of N, but this is a poisson process
        N = max(0,round(np.random.normal(loc=N_exp,scale = np.sqrt(N_exp))))
        print(f'{N_exp:.2f}-->{N}')
        new_max_areas = np.random.lognormal(mean=np.log(self.average_spot_area/MSH),sigma=sigma,size=N)*MSH
        # now assign lat and lon (dist approx from 2017ApJ...851...70M)
        hemi = np.random.choice([-1,1],size = N)
        lat = np.random.normal(15,5,size=N)*hemi*u.deg
        lon = np.random.random(size=N)*360*u.deg
        
        penumbra_teff = Teff_star*0.8
        umbra_teff = penumbra_teff*0.8
        
        spots = []
        for i in range(N):
            spots.append(StarSpot(lat[i],lon[i],new_max_areas[i],starting_size,umbra_teff,penumbra_teff))
        return tuple(spots)