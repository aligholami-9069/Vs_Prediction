# A committee machine approach for estimation of fast and slow shear wave velocities utilizing petrophysical logs

### By:
### Ali Gholami Vijouyeh<sup>a</sup>, Ali Kadkhodaie<sup>a</sup>, Mohammad Hassanpour Sedghi<sup>a</sup>, Hamed Gholami Vijouyeh<sup>b</sup>

<sup>a</sup> Department of Earth Science, Faculty of Natural Science, University of Tabriz, Tabriz, Iran

<sup>b</sup> Business Intelligence Developer, Sales Performance Management, DHL Supply Chain Management GmbH, Bonn, Germany

---

## Content


### __Main__

* #### __correlation_plot.m__
    Calculates and shows the correlation between different fields of data.
* #### __create_data.m__
    Creates training and test data from selected features of main data. The result is saved in ".\01 Data\test_data\\" folder.
* #### __fuzzy_logic.m__
    Implements Fuzzy Logic algorithm and stores the results in ".\03 Results\Fuzzy Logic\\".
* #### __neural_network.m__
    Implements Neural Network algorithm. The script reads the proper network from ".\03 Results\Neural Network\Net\\" and saves the result in ".\03 Results\Neural Network\\".
* #### __neuro_fuzzy.m__
    Implements Neuro-Fuzzy algorithm. It reads the network from ".\03 Results\Neuro Fuzzy\Net\\" and stores the result in ".\03 Results\Neuro Fuzzy\\" folder.
* #### __optimization.m__
    This file contains all optimization algorithms in the article, which are "GA", "Simulated Annealing", and "Ant Colony". The result is saved in ".\03 Results\Optimization\\"


### __Parameter Tunning__

* #### __param_tunning_aco.m__
    This file has been used to parameter tunning for ACO optimization algorithm. The result is stored in ".\03 Results\Optimization\ACO\\" folder.
* #### __param_tunning_ga.m__
    As above script but tunes the parameters for GA. It stores the result in ".\03 Results\Optimization\GA\\" folder.
* #### __param_tunning_sa.m__
    This script tunes parameter for Simulated Annealing optimization algorithm. The results is stored in ".\03 Results\Optimization\SA\\" folder.