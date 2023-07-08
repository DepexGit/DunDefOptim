# DunDefOptim

Equipment optimization tool for Dungeon Defenders 1

## Installation
1. Install [Python 3](https://www.python.org/)
2. Install [pip](https://pypi.org/project/pip/)
3. Download this repository
4. Install requirements: `pip install -r requirements.txt`
5. Set your save file location by modifying *INPUT_FILE_PATH* in *src/optim_targets.py*. It should usually be enough to add the path to your Steam root directory containing your Dungeon Defenders installation to the first line.

## Usage
Optimal equipment is determined by maximizing the weighted sum of all stats. E.g.: If your weights are 1.0 for tower damage and 0.5 for 
tower rate and 0 for every other stat, then an item with 300 tower hp, 200 tower damage and 100 tower rate would get a score of 300 * 0 + 
200 * 1 + 100 * 0.5 = 250. You can specify one set of weights for every character whose gear you want to optimize by modifying 
*OPTIM_TARGETS* in *src/optim_targets.py*. Weights for stats you did not specify are automatically set to 0. When running the 
optimization, no item can the assigned to more than one character. The order of your characters in *OPTIM_TARGETS* determines their 
priority.

These two steps have to be repeated every time you obtain new relevant items:

1. Export your ranked save file to open mode
2. Decompress and parse the save file by running `python optimizer.py update`

Now you can either run the optimization:

`python optimizer.py optimize`

or print all your equipment:

`python optimizer.py print`

Run `python optimizer.py --help` for more options.

## Issues
Item types are determined by their id string in the save file. This works well for all armors, because their id strings all follow a simple pattern.
Other types of equipment, especially accessories are more difficult to detect and require knowledge of their specific id strings. Since there are
many items obtainable in the games which i do not own, it is likely that some of those id strings are missing. If this is the case and you have
such an item, you will get the exception *Unknown equipment type with id string abc* when running *update*. To fix this, simply add the id 
string to the appropriate variable in *src/consts.py*. Please also open an issue here on GitHub and let me know about the missing item.
