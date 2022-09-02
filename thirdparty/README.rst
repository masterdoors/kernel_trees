Patch to the Scikit-learn 0.22.X to work with Kernel Forests.

Differences:

- LibLinear primary and dual solvers consider the sample_weights paremeter.
- Trained SVC models do not store support vectors. User should provide original train dataset instead. (It is better in terms of memory when you have ~ 15K trained SVMs in one Kernel Forest)..
