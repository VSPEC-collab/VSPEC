## VSPEC

VSPEC is a tool designed to create time-series datasets combining spectroscopic light curves incorporating stellar variability with phase curves of M-star planets utilizing GCM's. It produces reflected light...

### Overview

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

### Directories and Files

The code is made up of 4 main programs:
* StarBuilder.py
* PlanetBuilder.py
* SpectraBuilder.py
* PlotBuilder.py

The code is executed based on a user-specified/curated config file:
* Config files are stored in `/Configs`
* Users can create their own configs based on example the template provided in the repository (ProxCenTemplate)
* The code also relies on a GCM config file which can be found in/uploaded to `/Configs/GCMs/`
* Users are free to create new configs for the star based on the template or introduce new GCM configs

All data are created when the code is run, based on the chosen stellar config file:
* A folder with the user-defined name of the star (based on config) is created.
* This folder contains two major sub-directories: Data saves all necessary data arrays and Figures stores all produced images/plots.

### Using the Code

The Code is split into 4 main executable programs, all named ____Builder.py, that you run individually and, to start, in this order.

1. StarBuilder.py
   - Creates the variable star model with spots and faculae, must be run first as the program needs a star.
   - Calculates the coverage fractions of the photosphere, spots, and faculae for each phase of the star.
   - It also bins the supplied stellar flux models (NextGen stellar dataset by default) to a resolving power specified in the config.
2. PlanetBuilder.py
   - Calls the Globes application of the Planetary Spectrum Generator web-tool.
   - Run second, after building th star; this produces a planet for the program.
   - The program sends information based on the user-defined config file including stellar and planetray parameters to PSG, starting with a General Circulation Model, and returns theoretical flux spectra of the planet's reflected and thermal flux.
3. SpectraBuilder.py
   - Run third; it needs saved planetary spectra and stellar spectra.
   - Uses the output of the StarBuilder.py and PlanetBuilder.py to create a synthetic timeseries of flux data that applies the planet flux from PSG to the star model created in StarBuilder.py, so the data we see is of the planet's flux as if it were revolving around the newly created, variable star.
4. PlotBuilder.py
   - Creates many plots displaying lightcurves, stellar flux output, planet flux output, planet thermal flux output, total system output, etc. across a timeseries.

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/cameronkelahan/VSPEC/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
