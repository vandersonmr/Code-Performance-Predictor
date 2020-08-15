# GNN Based Code Performance Predictor 

## Performance predictor based on Graph Neural Networks (GNN)

We use a GNN to try to learn to predict the performance difference of an application compiled with two set of optimizations using the LLVM compiler. The dataset used is described bellow. 

For that, we use the network described in the following scheme:

![alt text](https://github.com/vandersonmr/Code-Performance-Predictor/blob/master/network.png)

The implementation of the network is in GNN.py and it uses the Spektral framework (https://github.com/danielegrattarola/spektral)

To test it:

```shell
pip3 install tensorflow spektral matplotlib sklearn numpy
python3 GNN.py
```

This implementation (and Spektral) requires Tensorflow 2.

## The dataset

In data/ there is a serialized NumPy array named small150.npz. This file includes three NumPy arrays, one for pair of adjacent matrixes representing Control Flow Graphs (CFGs) for the same application compiled twice with different optimization plains, the second contains features extract from the code binary of each node counting the number of instructions of it and, the last contains the relative difference of performance from the two pairs of CFGs. They have the following shapes:

* Pairs of CFGs: (2, 25500, 150, 150)
* CFGs features: (2, 25500, 150, 94)
* Relative speedup: (25500, 1)

There are 25500 pairs of CFGs collected from a set of programs compiled with LLVM with different optimization plains. 

The goal is to predict the performance difference of an application compiled with two different compilation plains using information about the amount of x86_64 final total of instruction and the CFG instruction. 


## License

This code is under MIT license and the dataset in Creative Commons Attribution 4.0 International.
