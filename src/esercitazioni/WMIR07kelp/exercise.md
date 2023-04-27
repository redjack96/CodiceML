# Exercise

These exercises can be executed directly in JAVA or from the JSON files:
1. Apply the normalization step to all the kernels defined in the Java file
   discussed in class.
2. Fix all the hyperparameters _(C, degree, etc.)_ and find the best kernel
   function in terms of final Accuracy between _linear, polynomial and
   tree kernel_.
3. Ablation study: combine all the kernels to create all the permutations as
   shown in class (e.g. _bow_tree_kernel)_ to find the best combination in
   terms of Accuracy.

These exercises should be executed only with JAVA or Python (but you should create
   the kelp file from Python with that precise syntax, see Kelp):
1. Add more features (_representations_) to every example:
   a. Lemma
   b. POS tagging
   c. Triple relations such as <_subject_, ran, Alice>
2. Find the best kernel function and hyperparameters using all the features
   (_representations_).