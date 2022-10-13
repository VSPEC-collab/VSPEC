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
    def __init__(self,Teff,radius,period,spots,faculae,name='',distance = 1*u.pc,resolution = {'lat':500,'lon':1000}):
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
        assert isinstance(faculae,FaculaCollection)
        self.faculae = faculae
        self.resolution = resolution
        self.map = self.get_pixelmap(Nlat=self.resolution['lat'],Nlon=self.resolution['lon'])
        
        self.spot_generator = SpotGenerator(500*MSH,200*MSH,coverage=0.15)
        self.fac_generator = FaculaGenerator()
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
        self.spots.add_spot(spot)
        self.map = self.get_pixelmap(self.resolution['lat'],self.resolution['lon'])
    def add_fac(self,facula):
        """add facula(e)
        Add a facula or faculae

        Args:
            facula (Facula or series of Faculae): Facula objects to add
        
        Returns:
            None
        """
        self.faculae.add_faculae(facula)
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

        int_map, map_keys = self.faculae.map_pixels(latgrid,longrid,self.map,self.radius,self.Teff)

        Teffs = np.unique(self.map)
        data = {}
        # spots and photosphere
        for teff in Teffs:
            pix = self.map == teff
            pix_sum = ((pix.astype('float32') * cos_c * jacobian)[int_map==0]).sum()
            data[teff] = pix_sum
        for i in map_keys.keys():
            facula = self.faculae.faculae[i]
            angle = 2 * np.arcsin(np.sqrt(np.sin(0.5*(facula.lat - sub_obs_coords['lat']))**2
                                + np.cos(facula.lat)*np.cos(sub_obs_coords['lat']) * np.sin(0.5*(facula.lon - sub_obs_coords['lon']))**2 ))
            frac_area_dict = facula.fractional_effective_area(angle)
            loc = int_map == map_keys[i]
            pix_sum = (loc.astype('float32') * cos_c * jacobian).sum()
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
        T (astropy.units.quantity.Quantity [time]): facula lifetime
        growing (bool): whether or not the facula is still growing
        floor_threshold (astropy.units.quantity.Quantity [length]): facula radius under which the floor is no longer visible
    
    Returns:
        None
    """
    def __init__(self,lat,lon,Rmax,R0,Teff_floor,Teff_wall,T,growing=True,floor_threshold = 20*u.km,Zw=100*u.km):
        assert u.get_physical_type(lat) == 'angle'
        assert u.get_physical_type(lon) == 'angle'
        self.lat = lat
        self.lon = lon
        assert u.get_physical_type(Rmax) == 'length'
        assert u.get_physical_type(R0) == 'length'
        assert u.get_physical_type(Zw) == 'length'
        self.Rmax = Rmax
        self.current_R = R0
        self.Zw = Zw
        assert u.get_physical_type(Teff_floor) == 'temperature'
        assert u.get_physical_type(Teff_wall) == 'temperature'
        self.Teff_floor = Teff_floor
        self.Teff_wall = Teff_wall
        assert u.get_physical_type(T) == 'time'
        assert isinstance(growing,bool)
        self.lifetime = T
        self.is_growing = growing
        assert u.get_physical_type(floor_threshold) == 'length'
        self.floor_threshold = floor_threshold
    
    def age(self,time):
        """age
        progress the development of the facula by an amount of time
        
        Args:
            time (astropy.units.quantity.Quantity [time]): amount of time to progress facula
        
        Returns:
            None
        """
        assert u.get_physical_type(time) == 'time'
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
            return {self.Teff_floor:0.0,self.Teff_wall:1.0}
        else:
            x = np.linspace(0,1,N) * self.current_R #distance from center along azmuth of disk
            h = np.sqrt(self.current_R**2 - x**2) # effective radius of the 1D facula approximation
            critical_angles = np.arctan(2*h/self.Zw)
            Zeffs = np.sin(angle)*np.ones(N) * self.Zw
            Reffs = np.cos(angle)*h*2 - self.Zw * np.sin(angle)
            no_floor = critical_angles < angle
            Zeffs[no_floor] = h[no_floor]*np.cos(angle)
            Reffs[no_floor] = 0
            
            return {self.Teff_wall: np.trapz(Zeffs,x),self.Teff_floor: np.trapz(Reffs,x)}
    
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
    def map_pixels(self,latgrid,longrid,star_rad):
        """map pixels
        
        """
        lat0 = self.lat
        lon0 = self.lon
        rad = self.angular_radius(star_rad)
        r = 2* np.arcsin(np.sqrt(np.sin(0.5*(lat0-latgrid))**2
                         + np.cos(latgrid)*np.cos(lat0)*np.sin(0.5*(lon0 - longrid))**2))
        pix_in_fac = r<=rad
        return pix_in_fac

class FaculaCollection:
    """Facula Collection
    Containter class to store faculae
    
    Args:
        *faculae (tuple): series of faculae objects
    
    Returns:
        None
    """
    def __init__(self,*faculae):
        self.faculae = tuple(faculae)
    def add_faculae(self,facula):
        """add facula(e)
        Add a facula or faculae

        Args:
            facula (Facula or series of Facula): Facula object(s) to add
        
        Returns:
            None
        """
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
            if (facula.current_R <= facula.current_R/np.e**2) and (not facula.is_growing):
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
    def map_pixels(self,latgrid,longrid,pixmap,star_rad,star_teff):
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
        int_map = np.zeros(shape=pixmap.shape,dtype='int8')
        map_dict = {}
        for i in range(len(self.faculae)):
            facula = self.faculae[i]
            pix_in_fac = facula.map_pixels(latgrid,longrid,star_rad)
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
    def __init__(self,R_peak = 100*u.km, R_HWHM = 50*u.km,
                 T_peak = 3.2*u.hr, T_HWHM = 2.7*u.hr,coverage=0.01,dist = 'even'):
        assert u.get_physical_type(R_peak) == 'length'
        assert u.get_physical_type(R_HWHM) == 'length'
        assert u.get_physical_type(T_peak) == 'time'
        assert u.get_physical_type(T_HWHM) == 'time'
        radius_unit = u.km
        lifetime_unit = u.hr
        self.R0 = np.log10(R_peak/radius_unit)
        self.sig_R = np.log10((R_peak + R_HWHM)/radius_unit) - self.R0
        self.T0 = np.log10(T_peak/lifetime_unit)
        self.sig_T = np.log10((T_peak + T_HWHM)/lifetime_unit) - self.T0
        assert isinstance(coverage,float)
        self.coverage = coverage
        self.dist = dist
        
    def get_floor_teff(R,Teff_star):
        """Get floor Teff
        Get the Teff of the faculae floor based on the radius and photosphere Teff
        Based on K. P. Topka et al 1997 ApJ 484 479
        
        Args:
            R (astropy.unit.quantity.Quantity [length]): radius of the facula[e]
            Teff_star (astropy.unit.quantity.Quantity [temperature]): effective temperature of the photosphere
        
        Returns:
            (astropy.unit.quantity.Quantity [temperature]): floor temperature of faculae
        """
        d_teff = np.zeros(len(R))
        reg = R <= 150*u.km
        d_teff[reg] = -1 * u.K * R[reg]/u.km/5
        reg = (R > 150*u.km) & (R <= 175*u.km )
        d_teff[reg] = 510 * u.K - 18*R[reg]/5/u.km*u.K
        reg = (R > 175*u.km)
        d_teff[reg] = -4*u.K*R[reg]/7/u.km - 20 * u.K
        
        return d_teff + Teff_star
    
    def get_wall_teff(R,Teff_floor):
        """Get wall Teff
        Get the Teff of the faculae wall based on the radius and floor Teff
        Based on K. P. Topka et al 1997 ApJ 484 479
        
        Args:
            R (astropy.unit.quantity.Quantity [length]): radius of the facula[e]
            Teff_floor (astropy.unit.quantity.Quantity [temperature]): effective temperature of the cool floor
        
        Returns:
            (astropy.unit.quantity.Quantity [temperature]): wall temperature of faculae
        """
        return Teff_floor + R/u.km * u.K + 125*u.K
        
        
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
                        * time/(10**self.T0 * self.lifetime_unit)).to(u.Unit(''))
        # N_exp is the expectation value of N, but this is a poisson process
        N = max(0,round(np.random.normal(loc=N_exp,scale = np.sqrt(N_exp))))
        print(f'{N_exp:.2f}-->{N}')
        mu = np.random.normal(loc=0,scale=1,size=N)
        max_radii = 10**(self.R0 + self.sig_R * mu) * self.radius_unit
        lifetimes = 10**(self.T0 + self.sig_T * mu) * self.lifetime_unit
        starting_radii = max_radii / np.e**2
        lats = None
        lons = None
        if self.dist == 'even':
            x = np.linspace(-90,90,180,endpoint=False)
            p = np.cos(x)
            lats = (np.random.choice(x,p=p/p.sum(),size=N) + np.random.random(size=N)) * u.deg
            lon = np.random.random(size=N) * 360 * u.deg
        else:
            raise NotImplementedError(f'{self.dist} has not been implemented as a distribution')
        teff_floor = self.get_floor_teff(max_radii,Teff_star)
        teff_wall = self.get_wall_teff(max_radii,teff_floor)
        new_faculae = []
        for i in range(N):
            new_faculae.append(Facula(lats[i], lons[i], max_radii[i], starting_radii[i], teff_floor[i],
                                     teff_wall[i], growing=True, floor_threshold=20*u.km, Zw = 100*u.km))
        return tuple(new_faculae)