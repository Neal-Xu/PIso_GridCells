# THE PRINCIPLE OF ISOMORPHISM: A THEORY OF POPULATION ACTIVITY IN GRID CELLS AND BEYOND

## Environment setup

We provide an `environment.yml` file to reproduce our experimental environment.  
To create and activate the environment, run:

```bash
conda env create -f environment.yml
conda activate TIS
```

This will install all required dependencies, including PyTorch, NumPy, SciPy, scikit-learn, matplotlib, and UMAP.

Note:

- The environment was tested with Python 3.10 and CUDA 12.4 (for GPU training).

## Train
```
python train_model.py --rho 1 --size 0.5 --run 0
python train_model.py --rho 0.8 --size 0.84 --run 1
```
## Analyze the network
```
python representation.py --rho 1 --size 0.5 --run 0
python representation.py --rho 0.8 --size 0.84 --run 1
```

## Systematic training
```
python param_search.py
python run_noise_sweep.py
```

## Systematic analysis
```
python analyze_cell.py
```

## Acknowledgments

The implementation was developed primarily by [Fei Song](https://github.com/Sophycountry), based on the framework of Pettersen et al. (2024) [Github repo](https://github.com/bioAI-Oslo/GridsWithoutPI), with extensions and modifications by Maoshen Xu, licensed under the MIT License. We thank the authors for releasing their code. 
On top of their framework, we explicitly enforced toroidal population activity, introduced the torus-size regularization and systematically explored its interaction with conformal isometry loss.

