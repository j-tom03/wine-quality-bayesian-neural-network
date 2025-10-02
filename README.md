# Wine Quality prediction using Bayesian Neural Networks

Applying different posterior approximation methods, such as VI and MCMC, and then training a BNN to predict wine quality. 

The report then outlines a comparison.

The project contains:
* `models.py` -- the main program containing all of the BNN, VI and MCMC code
* `preprocessing.py` -- the program containing all preprocessing methods
* `plots.py` -- the program containing all functions used to plot graphs
* `prelim_plots.py` -- the program used to plot all plots related to the raw data
* `requirements.txt` -- a text file containing all python packages required for the program
* `Data` -- a folder containing two CSV files of the data
* `Report.pdf` -- a written report explaining the methodology and comparing VI and MCMC using empirical data
* `Results.csv` -- containing all the training results

Note: You will need to add a folder `plots` in the root directory for the plots to store

## Running the Models
```python
python models.py
```

## Requirements
- pandas
- matplotlib
- seaborn
- numpy
- imblearn
- scikit-learn
- torch
- pyro-ppl
- tqdm


## Dataset

The dataset can be found [here](https://archive.ics.uci.edu/dataset/186/wine+quality)


I completed this project whilst undertaking the COM3023 module at the University of Exeter.

Grade: 70/100 (graded only on report not code)
