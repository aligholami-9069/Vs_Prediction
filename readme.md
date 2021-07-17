# A committee machine approach for estimation of fast and slow shear wave velocities utilizing petrophysical logs

### By:
### Ali Gholami Vijouyeh<sup>a</sup>, Ali Kadkhodaie Ilkhchi<sup>a</sup>, Mohammad Hassanpour Sedghi<sup>a</sup>, Hamed Gholami Vijouyeh<sup>b</sup>

<sup>a</sup> Department of Earth Science, Faculty of Natural Science, University of Tabriz, Tabriz, Iran

<sup>b</sup> Business Intelligence Developer, Sales Performance Management, DHL Supply Chain Management GmbH, Bonn, Germany

---

## Content
* ### __correlation_plot.m__
    Calculates and shows the correlation between different fields of data.
* ### __create_data.m__
    Creates training and test data from selected features of main data. The result is saved in ".\Data\trn_tst\\" folder.
* ### __fuzzy_logic.m__
    Implements Fuzzy Logic algorithm and stores the results in ".\Results\Fuzzy Logic\\".
* ### __neural_network.m__
    Implements Neural Network algorithm. The script reads a proper network from ".\Results\Neural Network\Net\\" and saves the result in ".\Results\Neural Network\\".
* ### __neuro_fuzzy.m__
    Implements Neuro-Fuzzy algorithm. It reads the network from ".\Results\Neuro Fuzzy\Net\\" and stores the result in ".\Results\Neuro Fuzzy\\" folder.
* ### __optimization.m__
    This file contains all optimization algorithms in the article, which are "Simple Averaging", "GA", "Simulated Annealing", "Ant Colony", and "Total Averaging". The result is saved in ".\Results\Optimization\\"
* ### __script_plot.m & script_plot_2.m__
    These two scripts plot the final results and store them in ".\Results\Plots\\" folder.


