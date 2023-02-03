#
# TEST GEOMETRY
# --------------
# Objective 1
# Create a gif with some random parameters

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy import units as u
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from os import system


from os import chdir, remove


from VSPEC import ObservationModel
from VSPEC.helpers import isclose
from pathlib import Path




def make_fig(phase,sub_obs,filename):
    planet_fig = geo.plot(phase)
    planet_fig.canvas.draw()
    star_fig = model.star.plot_spots(sub_obs,sub_obs)
    star_fig.canvas.draw()

    gs = gridspec.GridSpec(16, 16,wspace=0,hspace=0,figure=plt.figure(figsize=(5,5)))
    ax1 = plt.subplot(gs[:8, :])
    im=np.array(planet_fig.canvas.renderer.buffer_rgba())
    ax1.imshow(im)
    ax1.axis('off')
    del planet_fig


    ax3 = plt.subplot(gs[8:, :])
    im=np.array(star_fig.canvas.renderer.buffer_rgba())
    ax3.imshow(im)
    ax3.axis('off')
    del star_fig

    plt.savefig(filename,facecolor='w',dpi=150)

def make_gif(plan,filename):
    temp_path = Path('temp/')
    if not temp_path.exists():
        temp_path.mkdir()
    N_images = len(plan['phase'])
    for i in tqdm(range(N_images),desc='Creating images', total=N_images):
        phase = plan['phase'][i]
        lat = plan['sub_obs_lat'][i]
        lon = plan['sub_obs_lon'][i]
        coords = {
            'lat':lat,
            'lon':lon
        }
        fname=f'phase{str(i).zfill(3)}.png'
        make_fig(phase,coords,temp_path / fname)
        plt.close('all')
    frames = []
    for i in tqdm(range(N_images),desc='Creating gif', total=N_images):
        fname=f'phase{str(i).zfill(3)}.png'
        frames.append(Image.open(temp_path / fname))
    frame_one = frames[0]
    frame_one.save(filename, format="GIF", append_images=frames, save_all=True, duration=70, loop=False)
    system('rm temp/*')



if __name__ in '__main__':
    CONFIG_FILENAME = 'test_geometry_2.cfg'
    WORKING_DIRECTORY = Path(__file__).parent
    CONFIG_PATH = WORKING_DIRECTORY / CONFIG_FILENAME

    chdir(WORKING_DIRECTORY)


    model = ObservationModel(CONFIG_FILENAME)


    model.build_star()
    model.warm_up_star(0*u.day,0*u.day)

    geo = model.get_observation_parameters()
    plan = model.get_observation_plan(geo)
    
    make_gif(plan,'geometry.gif')
    # geo.plot(0*u.deg)
    # i=0
    # phase = plan['phase'][i]
    # lat = plan['sub_obs_lat'][i]
    # lon = plan['sub_obs_lon'][i]
    # coords = {
    #     'lat':lat,
    #     'lon':lon
    # }
    # make_fig(phase,coords, 'phase1.png')
