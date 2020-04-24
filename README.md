Code for creating and training an SVM from scratch (i.e. just using numpy). Includes implementations for Log barrier Interior Point method with feasible start Newton and Sequential Minimal Optimisation (SMO). An example Jupyter Notebook training an SVM with both optimisation methods on the MNIST dataset is included. Note, code is for a binary SVM classifier, and the SMO implementation does not use advanced heuristics for picking the order of dual variables to optimise. 

Both optimisations optimise the dual objective, so kernelisation is easily permitted. I've included examples of a Gaussian and polynomial kernel in the kernels file.

References I found very helpful:
http://cs229.stanford.edu/materials/smo.pdf (SMO implementation heavily relies on this pseudocode)
http://cs229.stanford.edu/notes/cs229-notes3.pdf
https://mml-book.github.io/book/mml-book.pdf
