#include <iostream>
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

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
    BDCSVD<MatrixXd, ComputeFullU | ComputeFullV> svd(X0.transpose() * Y0);
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
        s = ((X0 * R).transpose() * Y0).trace() / X0.squaredNorm();
    }

    // Producing the aligned X
    MatrixXd aligned_X = s * X0 * R;
    aligned_X.rowwise() += Y_mean.transpose();

    // Translation component (optional)
    MatrixXd translation_vec = Y_mean.transpose() - s * (X_mean.transpose() * R);

    return {aligned_X, R, s, translation_vec};
}


std::tuple<MatrixXd, std::vector<MatrixXd>, std::vector<double>> GPA(
    const std::vector<MatrixXd>& X_list_input, 
    bool scale = true, 
    double epsilon = 1e-4)
{
    int m = X_list_input.size();
    std::vector<MatrixXd> X_list(m);

    // STEP 1: Copy and center the matrices
    double fro_sum = 0;
    for (int i = 0; i < m; ++i) {
        X_list[i] = X_list_input[i];
        VectorXd X_mean = X_list[i].colwise().mean();
        X_list[i].rowwise() -= X_mean.transpose();
        fro_sum += X_list[i].squaredNorm();
    }

    double var_lambda = std::sqrt(static_cast<double>(m) / fro_sum);
    std::vector<double> scaling_factors(m, var_lambda);

    for (int i = 0; i < m; ++i) {
        X_list[i] *= var_lambda;
    }

    MatrixXd Y_consensus = X_list[0];

    // STEP 2: Align the initial matrices to the consensus
    for (int i = 1; i < m; ++i) {
        X_list[i] = std::get<0>(CPA(X_list[i], Y_consensus, false, false));
    }

    Y_consensus = MatrixXd::Zero(Y_consensus.rows(), Y_consensus.cols());
    for (int i = 0; i < m; ++i) {
        Y_consensus += X_list[i];
    }
    Y_consensus /= m;

    double Sr = m * (1 - Y_consensus.squaredNorm());

    // STEP 3: Iterative alignment
    while (true) {
        // Align all the matrices to the current consensus
        for (int i = 0; i < m; ++i) {
            X_list[i] = std::get<0>(CPA(X_list[i], Y_consensus, false, false));
        }

        // STEP 4: Skipped in this opimized version
        // See "Практические аспекты"

        // STEP 5: Optional scaling
        if (scale) {
            for (int i = 0; i < m; ++i) {
                double numerator = (X_list[i] * Y_consensus.transpose()).trace();
                double denominator = X_list[i].squaredNorm() * Y_consensus.squaredNorm();
                double change_factor = std::sqrt(numerator / denominator);
                double s_i_star = scaling_factors[i] * change_factor;
                X_list[i] *= s_i_star / scaling_factors[i];
                scaling_factors[i] = s_i_star;
            }
        }

        // Update consensus
        MatrixXd Y_consensus_new = MatrixXd::Zero(Y_consensus.rows(), Y_consensus.cols());
        for (int i = 0; i < m; ++i) {
            Y_consensus_new += X_list[i];
        }
        Y_consensus_new /= m;

        double Sr_new = Sr - m * (Y_consensus_new.squaredNorm() - Y_consensus.squaredNorm());
        Y_consensus = Y_consensus_new;

        // STEP 6: Check for the termination condition
        if (Sr - Sr_new < epsilon) break;

        Sr = Sr_new;
    }

    // STEP 7: Exit the procedure
    return {Y_consensus, X_list, scaling_factors};
}