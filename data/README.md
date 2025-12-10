gpatp1.txt, ... , gpatph92.txt are the results in the format <m (input size), n (matrix dimension), k (iteration count), time elapsed while in GPA>. The number in the filename before the file format is the number of processes used in the corresponding experiment.

eig1.txt, ... , eig8.txt are the results in the format <m (input size), n (matrix dimension), k (iteration count), time elapsed while in GPA> with additional Eigen information printed showing the time elapsed while in GPA for every Eigen OpenMP configuration. The number in the filename before the file format is the number of OpenMP threads available to Eigen to be used internally.

gpa_iter_0.txt, gpa_iter_1.txt are the results showing the number of iterations for GPA with scaling disabled, and with d = 3. The format is as above.
