## Approximate zero-crossing feature for EEG 

A new interpretable, highly discriminative and low complex feature for EEG and iEEG seizure detection

Paper: „Approximate Zero-Crossing: A new interpretable, highly discriminative and low-complexity feature for EEG and iEEG seizure detection“, Renato Zanetti, Una Pale, Tomas Teijeiro, David Atienza Alonso
Journal of Neural Engineering, https://iopscience.iop.org/article/10.1088/1741-2552/aca1e4 


## Licence and requirements 

License: LGPL

We have used Anaconda toolset, including the following packages (original and added later):

- Python 					(v3.9.7)
- numpy                     (v1.20.3)
- pandas                    (v1.3.5)
- scikit-learn              (v1.0.2)
- scipy                     (v1.7.3)
- seaborn                   (v0.11.2)
- matplotlib                (v3.5.0)
- pyedflib                  (v0.1.25)
- antropy					(v0.1.4)
- PyWavelets				(v1.1.1)

## Scripts

1) `script_AZC_datasetPreProcessing.py`: **to be executed first**, including feature extraction and preparation for TSCV execution.
2) `script_AZC_FeatDivergence.py`: calculates the KL divergence and generate related plots.
3) `script_AZC_Classification.py`: used for seizure classification (TSCV approach). 
4) `script_AZC_ConsolidateResults.py`: consolidation of results and its plots.

All scripts include few expected user setups before execution (e.g., the path of the folder containing the datasets), also observing the parameters set in `parametersSetup.py`.

It's important to note that the scrpit `script_AZC_datasetPreProcessing.py` employs a pool of processing cores in case the variable **parallelize** is set. Similar behaviour can be achieved for `script_AZC_Classification.py` script, but in this case, you should launch the execution via **bash** using the `bash_run_python_variosPat.sh` script.

