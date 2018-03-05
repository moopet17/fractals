# fractals
Modular program to run differentiation and integration tasks following a statistical learning phase.

### Setup for Windows 10:

Follow [these Instructions from the University of Connecticut](https://psychologyit.uconn.edu/2017/09/20/instructions-for-installing-psychopy/). Please be aware that the [Anaconda instructions on the Psychopy website](http://psychopy.org/installation.html) were out of date and did not work. The specific steps that worked were: 

1. Install [Anaconda Python 2.7 version](https://www.anaconda.com/download/).
2. Open the Anaconda Prompt from the Start Menu (Windows command prompt should work too).
3. Create a new python 2.7 environment with the name psychopy that includes the following packages and their dependencies: psychopy pyglet wxpython pygame by copy/pasting `conda create -n psychopy --channel https://conda.anaconda.org/CogSci psychopy pyglet wxpython pygame python=2.7` into your command prompt, hit enter, confirm by hitting `y` and enter.
4. Activate the environment: `activate psychopy`
5. Install pandas: `pip install pandas`
6. "Change directories" (cd) into the folder containing the script file "fractals.py" by entering `cd <folder-path>`
7. Paste/type `python fractals.py` and hit enter
8. When you’re done working on it, enter `deactivate` to leave the environment.
