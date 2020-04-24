## SVM from Scratch

This repository includes code for training a kernelised SVM with only numpy (specifically does not use any optimization libraries for quadratic programming). 

Includes implementations for 
* Log barrier Interior Point method with feasible start Newton,
* Sequential Minimal Optimisation (SMO)
* Kernels (Gaussian and Polynomial)

Both optimisations optimise the dual objective, so kernelisation is easily permitted. This repo also includes an example Jupyter Notebook training an SVM with both optimisation methods on the MNIST dataset. 

Some notes: the code is for a binary SVM classifier, and the SMO implementation does not use advanced heuristics for picking the order of dual variables to optimise. 

References I found very helpful:
* http://cs229.stanford.edu/materials/smo.pdf (SMO implementation heavily relies on this pseudocode)
* http://cs229.stanford.edu/notes/cs229-notes3.pdf
* https://mml-book.github.io/book/mml-book.pdf
