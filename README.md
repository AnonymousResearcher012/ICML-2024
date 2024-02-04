# ICML-2024

This is the code supplementing the submission "Certified Invariant Polytope Training in Neural Controlled ODEs".

## Install `immrax` and other dependencies

First, navigate to the `immrax` folder, and follow the steps in its `README.md` to install it and its dependencies.

Finally, there are some extra dependencies to install to generate the figures:
```bash
pip install -r requirements.txt
```

## Basic Example

To run the basic example in Examples 4.1 and 4.8 and generate Figure 1, navigate into the example1 folder and run the script.
```bash
cd example1
python example1.py
```

## 2D Double Integrator With Nonlinarity and Disturbance
Navigate into the `xydoubleintegrator` folder.
```bash
cd xydoubleintegrator
```

### Compare pre-trained models
To compare the pretrained models and generate Figure 2, run the following:
```bash
python compare.py
```

### Train model
To train a new neural network controller, run the following:
```bash
python train.py
```
This can take some time to trace the function computing the gradient of the loss function. To change the loss function, navigate to line 121. If training without the invariance loss, the exit condition on line 161 may need to be changed (the network will otherwise train for 100000 iterations).

## Segway Model
Navigate into the `segway` folder.

### Visualize pre-trained model
To visualize the pretrained model and generate Figure 3, run the following:
```bash
python test.py
```

### Train model
To train a new neural network controller, run the following:
```bash
python train.py
```
This can take a long time to trace the full `immrax` stack to generate the function computing the gradient of the loss function. 
