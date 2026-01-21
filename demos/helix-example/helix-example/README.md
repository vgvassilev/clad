[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/compiler-research/helix-example/HEAD)
# Helix fitter example

## Goal

Particle tracking is an important part of the processing and analysis of data received from particle detectors, such as the Compact Muon Solenoid (CMS). Tracking is the step that determines the momentum of charged particles escaping from the collision point. It identifies individual particles by reconstructing their trajectories from points where charged particle “hits” were measured by the detector and interpreting them.[1] Due to the Lorentz force, charged particles move in a helical motion when affected by the magnetic field (neglecting other effects due to material interactions, etc). This means we can figure out a specific particle trajectory through the detector by fitting a helix function to data points in such a way that the distance from the data points and the helix would be minimized. In mathematical terms, we need to find optimal helix parameters by minimizing a loss function composed of the sum of least squared distances, thus giving the best estimation of these parameters. For this purpose we can use Clad to efficiently minimize the loss function. 

In this repository one can find the code containing one such helix fitter implementation.

## Content

Besides the main fitter code, there are a few more files:

- Documentation.pdf - a more in depth explanation of the code and methods used.
- Helix.ipynb - a Jupyter notebook containing the code and the documentation comments. Can be used to try out the code online but is considerably slower.
- Graph_from_file.ipynb - accompanies Helix.ipynb. Reads from the output.txt (produced by Helix.ipynb) and plots the results.


## Usage

One can compile the files with:

```
clang++ main.cc -o main -I /full/path/to/clad/include/ -fplugin=/full/path/to/lib/clad.so -I /full/path/to/kokkos-4.3.01/include
```
Running the code and plotting the output can be done by:

```
./main | python3 graph.py
```
