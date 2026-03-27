Fundation of Deep Learning Course --- Experiment 1

FNN

load.py is responsible for Dataloader, normalization (only for labels; optional) and restore normalized labels.

nets.py defines different networks used in below experiments.

Please run 
    1. train_1_layer_num.py for comparison among different depths of networks (1-layer, 2-layer and 4-layer; see nets.py for details). 
    2. train_2_learning_rate.py for comparison among different learning rates for a given 2-layer networks.
    3. train_3_activation_function.py for comparison among different activation functions (ReLU, Sigmoid, Tanh, LeakyReLU and Swish).

### Important!
train_model function defined in train_x_xxx_xxx.py will automatically save the weights of the best model (with the lowest loss on the validation set) to the path "save_dir/best_diabetes_xxx.pth" ("save_dir" can be modified in train_x_xxx_xxx.py).
and if the directory "save_dir" doesn't exist, it will be created automatically!

The paths in the original program are written in Linux conventions!
