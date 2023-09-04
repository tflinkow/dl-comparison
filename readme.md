# dl-comparison

## Reproducing the results from the paper

The script [`run.sh`](run.sh) contains the exact configurations from the paper and can be run to replicate the results.
Assuming you have a reasonably up-to-date LaTeX distribution installed, it will generate the tables displaying the results as a file named `tables.pdf` and the plots as a file named `result-plots.pdf`.

Searching for values of the logical weight λ can be performed by running the script [`lambda-search.sh`](lambda-search.sh) and specifying the number of epochs to train for.
Our experiments used 200 epochs, so  by running `bash lambda-search.sh 200` you will be able to replicate our exact results.
This script will generate the plots as a file named `lambda-search-plots.pdf` and output the optimal value of λ to the console.

Plots and tables generated from the data used in the paper can be found in the [`figures/`](figures) directory.

## Requirements

The experiments were run on `Python 3.10`.
The provided `requirements.txt` can be used to install the required packages using `pip install -r requirements.txt`, however, it will only install stable versions.
The exact nightly build versions used were `torch 2.1.0.dev20230715` and `torchvision 0.16.0.dev20230715`.