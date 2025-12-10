
#include <iostream>
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <mpi.h>
#include <random>

using namespace Eigen;

std::tuple<MatrixXd, MatrixXd, double, MatrixXd> CPA(
    const MatrixXd& X,
    const MatrixXd& Y,
    bool scale = true,
    bool translation = true)
{
    MatrixXd X0, Y0;
    VectorXd X_mean, Y_mean;

    if (translation) {
        // Compute column-wise mean
        X_mean = X.colwise().mean();
        Y_mean = Y.colwise().mean();

        // Center matrices
        X0 = X.rowwise() - X_mean.transpose();
        Y0 = Y.rowwise() - Y_mean.transpose();
    } else {
        X_mean = VectorXd::Zero(X.cols());
        Y_mean = VectorXd::Zero(Y.cols());
        X0 = X;
        Y0 = Y;
    }

    // Singular value decomposition (SVD) of X0^T * Y0
    BDCSVD<MatrixXd> svd(X0.transpose() * Y0, ComputeFullU | ComputeFullV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    MatrixXd R = U * V.transpose();

    // Reflection check
    if (R.determinant() < 0) {
        V.col(V.cols() - 1) *= -1;
        R = U * V.transpose();
    }

    // Scaling if needed
    double s = 1.0;
    if (scale) {
        s = ((X0 * R).array() * Y0.array()).sum() / X0.squaredNorm();
    }

    // Producing the aligned X
    MatrixXd aligned_X = s * X0 * R;
    aligned_X.rowwise() += Y_mean.transpose();

    // Translation component (optional)
    MatrixXd translation_vec = Y_mean.transpose() - s * (X_mean.transpose() * R);

    return {aligned_X, R, s, translation_vec};
}


std::tuple<MatrixXd, std::vector<MatrixXd>, std::vector<double>> GPA(
    const std::vector<MatrixXd>& X_list_local,
    int rows,
    int cols,
    bool scale = true,
    double epsilon = 1e-4)
{
    int local_count = X_list_local.size();
    std::vector<MatrixXd> X_list(local_count);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // STEP 1: Copy and center the matrices
    double local_fro_sum = 0.0;
    #pragma omp parallel for reduction(+:local_fro_sum)
    for (int i = 0; i < local_count; ++i) {
        X_list[i] = X_list_local[i];
        VectorXd X_mean = X_list[i].colwise().mean();
        X_list[i].rowwise() -= X_mean.transpose();
        local_fro_sum += X_list[i].squaredNorm();
    }

    double global_fro_sum = 0.0;
    int m_total = 0;   
    MPI_Allreduce(&local_fro_sum, &global_fro_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &m_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);    

    double var_lambda = std::sqrt(static_cast<double>(m_total) / global_fro_sum);

    for (int i = 0; i < local_count; ++i) {
        X_list[i] *= var_lambda;
    }

    std::vector<double> scaling_factors(local_count, var_lambda);
    MatrixXd Y_consensus;
    if (local_count > 0) {
        Y_consensus = X_list[0]; // placeholder
    } else {
        Y_consensus = MatrixXd::Zero(rows, cols); // placeholder
    }

    // Make the consensus the first form from the rank 0 process
    MPI_Bcast(Y_consensus.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // STEP 2: Align the initial matrices to the consensus
    for (int i = 0; i < local_count; ++i) {
        X_list[i] = std::get<0>(CPA(X_list[i], Y_consensus, false, false));
    }

    // compute the global consensus
    MatrixXd local_sum = MatrixXd::Zero(rows, cols);
    for (int i = 0; i < local_count; ++i) {
         local_sum += X_list[i];
    }

    MatrixXd global_sum = MatrixXd::Zero(rows, cols);
    MPI_Allreduce(local_sum.data(), global_sum.data(), rows * cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Y_consensus = global_sum / m_total;

    double Sr = m_total * (1 - Y_consensus.squaredNorm());
  
    int iter_number = 1;

    // STEP 3: Iterative alignment
    while (true) {
        // Align all the matrices to the current consensus
        #pragma omp parallel for
        for (int i = 0; i < local_count; ++i) {
            X_list[i] = std::get<0>(CPA(X_list[i], Y_consensus, false, false));
        }

        // STEP 4: Skipped in this opimized version
        // See "Практические аспекты"

        // STEP 5: Optional scaling
        if (scale) {
            #pragma omp parallel for
            for (int i = 0; i < local_count; ++i) {
                double numerator = (X_list[i].array() * Y_consensus.array()).sum();
                double denominator = X_list[i].squaredNorm() * Y_consensus.squaredNorm();
                double change_factor = std::sqrt(numerator / denominator);
                double s_i_star = scaling_factors[i] * change_factor;
                X_list[i] *= s_i_star / scaling_factors[i];
                scaling_factors[i] = s_i_star;
            }
        }
        
        local_sum.setZero();
        for (int i = 0; i < local_count; ++i) {
            local_sum += X_list[i];
        }
               
        MPI_Allreduce(local_sum.data(), global_sum.data(), rows * cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        MatrixXd Y_consensus_new = global_sum / m_total;

        double Sr_new = Sr - m_total * (Y_consensus_new.squaredNorm() - Y_consensus.squaredNorm());
        Y_consensus = Y_consensus_new;

        // STEP 6: Check for the termination condition
        double local_err = Sr - Sr_new;
        MPI_Bcast(&local_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (local_err < epsilon) {
            if (rank == 0) {
                std::cout << iter_number << " ";
            }
            break;
        }
        
        ++iter_number;
        Sr = Sr_new;
        MPI_Bcast(&Sr_new, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // STEP 7: Exit the procedure
    return {Y_consensus, X_list, scaling_factors};
}

// Helper function in case reading from a file is needed
Eigen::MatrixXd readMatrix(const std::string &filename, int rows, int cols) {
    Eigen::MatrixXd mat(rows, cols);
    std::ifstream in(filename);
    if(!in) {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(1);
    }
    for(int i=0;i<rows;++i)
        for(int j=0;j<cols;++j)
            in >> mat(i,j);
    return mat;
}

// Helper function in case writing to a file is needed
void writeMatrix(const Eigen::MatrixXd& mat, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Cannot open file " << filename << " for writing.\n";
        return;
    }
    out << std::fixed << std::setprecision(8); // optional: match input precision
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            out << mat(i, j);
            if (j < mat.cols() - 1) out << " ";
        }
        out << "\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
     
    Eigen::setNbThreads(3);    
    int m = 192;
    while (m < 210) {
        int n = m;
        //while (n < 140) {
                int rows = n;
                int cols = n;
        
    		int chunk = (m + size - 1) / size;   // ceil division
    		int start = rank * chunk;
    		int end = std::min(start + chunk, m);
    		int local_count = std::max(0, end - start);    
		
		
    		std::vector<Eigen::MatrixXd> X_list_local(local_count);
    		std::mt19937 rng(rank + 1);
    		std::normal_distribution<double> dist(0.0, 1.0);
		
    		for (int i = 0; i < local_count; ++i) {
        		X_list_local[i] = MatrixXd(rows, cols);
        		for (int r = 0; r < rows; ++r) {
            		for (int c = 0; c < cols; ++c) {
                		X_list_local[i](r, c) = dist(rng);
            		}
        		}
    		}/*
    		if (rank == 0) {
        		std::cout << "Eigen reports max threads: " << Eigen::nbThreads() << std::endl;
    		}*/
                
                if (rank == 0) {
                    std::cout << m << " " << n << " ";
                }		

                double t_start = MPI_Wtime();
    		auto [Y_consensus, aligned_X_list, scaling_factors] = GPA(X_list_local, rows, cols, true, 1e-10);
    		double t_end = MPI_Wtime();
                double t_elapsed = t_end - t_start;
                
                if (rank == 0) {
                    std::cout << t_elapsed << std::endl;
                }

                //for (int i = 0; i < scaling_factors.size(); ++i) std::cout << rank << ": " << scaling_factors[i] << std::endl;
		
                //n += 10;
            //}
            m += 10;
        }


/*    if (rank == 0) {
    // Assume aligned_X_list has 6 matrices
        for (size_t i = 0; i < aligned_X_list.size(); ++i) {
            std::string filename = "aligned_X_" + std::to_string(i) + ".txt";
            writeMatrix(aligned_X_list[i], filename);
        }
    
        // Write consensus
        writeMatrix(Y_consensus, "Y_consensus.txt");
    
        for (int i = 0; i < scaling_factors.size(); ++i) {
            std::cout << scaling_factors[i] << ' ';
        }
    }
 */
    MPI_Finalize();
    return 0;

}

