There are 3 python files included: 

1)  function.py, which has all
    of the code to create models that simulate a function (part 1-1 of assignment)
    and graph the relevant data

2)  cifar.py, which has all of the code to train a model for CIFAR-10 task and graph the data (1-2) as well as
    the code to perform PCA and to graph the gradient norm (2-1 and 2-2)

3) random_labels.py which has all of the code to train the same CIFAR-10 model on random labels, and graph the accuracy and loss (3-1)

all 3 files require pytorch and python 3 to run.  They can all be run simply using "python file.py", or potentially "python3 file.py"
if you have python 2 and python 3 installed.  The graphs will be shown at the end of the training process one at a time; close one to open the next one. 

All training data needed will be downloaded by the script if its not already present.
PNG files of all figures in the report can be found in the figures folder.  A copy of the HW report is included for convenience.
