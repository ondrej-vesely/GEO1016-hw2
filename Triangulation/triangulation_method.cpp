/**
 * Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
 * https://3d.bk.tudelft.nl/liangliang/
 *
 * This file is part of Easy3D. If it is useful in your research/work,
 * I would be grateful if you show your appreciation by citing it:
 * ------------------------------------------------------------------
 *      Liangliang Nan.
 *      Easy3D: a lightweight, easy-to-use, and efficient C++
 *      library for processing and rendering 3D data. 2018.
 * ------------------------------------------------------------------
 * Easy3D is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 3
 * as published by the Free Software Foundation.
 *
 * Easy3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "triangulation.h"
#include "matrix_algo.h"
#include <easy3d/optimizer/optimizer_lm.h>


using namespace easy3d;


/// convert a 3 by 3 matrix of type 'Matrix<double>' to mat3
mat3 to_mat3(Matrix<double>& M) {
    mat3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}

/// convert a 1 by 3 matrix of type 'Matrix<double>' to vec3
vec3 to_vec3(Matrix<double>& M) {
    vec3 result = vec3(M(0, 0), M(0, 1), M(0, 2));
    return result;
}

/// convert M of type 'matN' (N can be any positive integer) to type 'Matrix<double>'
template<typename mat>
Matrix<double> to_Matrix(const mat& M) {
    const int num_rows = M.num_rows();
    const int num_cols = M.num_columns();
    Matrix<double> result(num_rows, num_cols);
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}

void normalize(std::vector<vec3>& points, mat3& ST) {

    // find centroid coordinate
    float centroid_x, centroid_y;

    centroid_x = 0;
    centroid_y = 0;

    for (int i = 0; i < points.size(); ++i) {
        centroid_x += points[i][0] / points[i][2];
        centroid_y += points[i][1] / points[i][2];
    }

    centroid_x = centroid_x / points.size();
    centroid_y = centroid_y / points.size();

    // create translation matrix
    mat3 T(1.0f);
    T.set_col(2, vec3(-centroid_x, -centroid_y, 1));

    // find scaling factors
    // get average distance of the points to the origin

    float total_dist = 0;

    // this is distance to centroid
    for (int i = 0; i < points.size(); ++i) {
        total_dist += sqrt(pow(points[i][0] - centroid_x, 2) + pow(points[i][1] - centroid_y, 2));
    }
  
    int avg_dist = total_dist / points.size();        // average distance

    // get scaling factor (dist * s = sqrt(2))
    float s = sqrt(2) / avg_dist;

    // Scaling Matrix S
    mat3 S = mat3::scale(s, s, 1);

    ST = S * T;

    // change input vector to its normalised version
    for (int i = 0; i < points.size(); ++i) {
        points[i] = ST * points[i];
    }
}

// Generate F matrix from normalized point pairs
Matrix<double> get_f_matrix(std::vector<vec3>& points_0n, std::vector<vec3>& points_1n) {
    
    // Generate W Matrix
    Matrix<double> W(points_0n.size(), 9, 0.0);
    for (int i = 0; i < points_0n.size(); ++i) {
        W.set_row({ points_0n[i][0] * points_1n[i][0], points_0n[i][1] * points_1n[i][0], points_1n[i][0],
                    points_0n[i][0] * points_1n[i][1], points_0n[i][1] * points_1n[i][1],
                    points_1n[i][1], points_0n[i][0], points_0n[i][1], 1 }, i);
    }

    // Decompose W with SVD
    int m = W.rows();
    int n = W.cols();
    Matrix<double> U_W(m, m, 0.0);
    Matrix<double> S_W(m, n, 0.0);
    Matrix<double> V_W(n, n, 0.0);
    svd_decompose(W, U_W, S_W, V_W);

    // Initialise and fill F matrix (F is a 3x3 matrix)
    Matrix<double> F(3, 3, 0.0);

    int k = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            F(i, j) = V_W(k, 8);
            k++;
        }
    }
    return F;
}

// Generate M matrix from R, t and K matrices
mat34 get_M_matrix(const mat3& R, const vec3& t, const mat3& K) {
    mat34 M;
    M.set_col(0, R.col(0));
    M.set_col(1, R.col(1));
    M.set_col(2, R.col(2));
    M.set_col(3, t);
    M = K * M;

    return M;
}

/// To determine how many points are in front of camera
/// for given relative camera pose
int points_in_front(const std::vector<vec3>& points_1, const mat3& R, const vec3& t, const mat3& K)
{
    int found = 0;
    mat34 M = get_M_matrix(R, t, K);

    for (vec3 p3 : points_1) {

        vec4 p4 = vec4{ p3[0], p3[1], p3[2], 1.0 };
        vec4 p4_proj = M * p4;
        if (p4_proj[2] > 0) { found++; }
    }
    return found;
}

vec3 get_3d_coordinates(vec4 M1, vec4 M2, vec4 M3, vec4 M1p, vec4 M2p, vec4 M3p, vec3 p1, vec3 p2) {
    mat4 A_(0.0);
    A_.set_row(0, p1[0] * M3 - M1);
    A_.set_row(1, p1[1] * M3 - M2);
    A_.set_row(2, p2[0] * M3p - M1p);
    A_.set_row(3, p1[1] * M3p - M2p);

    Matrix<double>A = to_Matrix(A_);

    Matrix<double> U_A(4, 4, 0.0);
    Matrix<double> S_A(4, 4, 0.0);
    Matrix<double> V_A(4, 4, 0.0);
    svd_decompose(A, U_A, S_A, V_A);

    std::vector<double> point_coord_h = V_A.get_column(3);

    vec3 point_coord(0, 0, 0);
    
    for (int i = 0; i < 3; i++) {
        point_coord[i] = point_coord_h[i] / point_coord_h[3];
    }
    
    return point_coord;
}

/**
 * TODO: Finish this function for reconstructing 3D geometry from corresponding image points.
 * @return True on success, otherwise false. On success, the reconstructed 3D points must be written to 'points_3d'.
 */
bool Triangulation::triangulation(
    float fx, float fy,                     /// input: the focal lengths (same for both cameras)
    float cx, float cy,                     /// input: the principal point (same for both cameras)
    const std::vector<vec3>& points_0,      /// input: image points (in homogenous coordinates) in the 1st image.
    const std::vector<vec3>& points_1,      /// input: image points (in homogenous coordinates) in the 2nd image.
    std::vector<vec3>& points_3d,           /// output: reconstructed 3D points
    mat3& R,                                /// output: recovered rotation of 2nd camera (used for updating the viewer and visual inspection)
    vec3& t                                 /// output: recovered translation of 2nd camera (used for updating the viewer and visual inspection)
) const
{
    // ---------   PART 0   ------------
    // --------- VALIDATION ------------
    // STEP 0.0 - CHECK IF INPUT POINTS ARE VALID

    if (points_0.size() < 8 || points_0.size() != points_1.size()) {
        std::cout << "\n" << "Input is not valid \n" << std::endl;
        return false;
    }
    else {
        std::cout << "\n" << "Input is valid \n" << std::endl;
    }


    // ---------              PART 1                ------------
    // --------- ESTIMATION OF FUNDAMENTAL MATRIX F ------------
    // STEP 1.0 - NORMALIZATION

    mat3 ST;               // for first camera - combined T and S for normalisation 
    mat3 ST_prime;         // for second camera

    std::vector<vec3> p_0n = points_0;
    std::vector<vec3> p_1n = points_1;

    // call normalise function defined above
    normalize(p_0n, ST);
    normalize(p_1n, ST_prime);


    // STEP 1.1 - LINEAR SOLUTION USING SVD ---------------------

    // call function to generate F matrix from input points
    Matrix<double> F = get_f_matrix(p_0n, p_1n);
    mat3 F_early = to_mat3(F);

    // STEP 1.2 - CONSTRAINT ENFORCEMENT (Based on SVD, Find the closest rank-2 matrix)

    // decompose F with SVD
    Matrix<double> U_F(3, 3, 0.0);
    Matrix<double> S_F(3, 3, 0.0);
    Matrix<double> V_F(3, 3, 0.0);
    svd_decompose(F, U_F, S_F, V_F);
    

    //recompose constrained F with rank(F)=2
    S_F(2, 2) = 0;
    Matrix<double> F_const = U_F * S_F * transpose(V_F);
    mat3 another_F = to_mat3(F_const);


    // STEP 1.3 - DENORMALIZATION -----------------------------------

    Matrix<double> ST_prime_double = to_Matrix(ST_prime);
    Matrix<double> ST_double = to_Matrix(ST);
    Matrix<double> F_den = ST_prime_double.transpose() * F_const * ST_double;
    mat3 F_ = to_mat3(F_den);


    // STEP 1.x - VERIFY --------------------------------------------

    /*
    std::cout << "ST matrix after" << ST << std::endl;
    std::cout << "ST_prime matrix after" << ST_prime << std::endl;
    std::cout << "W matrix after " << W << std::endl;
    std::cout << "F matrix " << F << std::endl;
    std::cout << "S_F matrix " << S_F << std::endl;
    std::cout << "constrained F matrix " << F_const << std::endl;
    */
    std::cout << "denormalised F matrix " << F_den << std::endl;


    // ------------------------ PART 2 ------------------------------------
    // --------- RECOVER RELATIVE POSE (R and t) FROM MATRIX F ------------

    // STEP 2.0 - CONSTRUCT K/K` MATRIX -----------------------------------

    // K Matrix with camera intrisic params
    // float fx, float fy,                     /// input: the focal lengths (same for both cameras)
    // float cx, float cy,                     /// input: the principal point (same for both cameras)

    mat3 K_(fx, 1,  cx,
            0,  fy, cy,
            0,  0,  1);
    Matrix<double> K = to_Matrix(K_);


    // STEP 2.1 - CALCULATE E MATRIX --------------------------------------

    // Essential matrix
    // E = K' * F * K

    Matrix<double> E = K.transpose() * F_den * K;
    mat3 E_ = to_mat3(E);


    // STEP 2.2 - FIND THE 4 CANDIDATE RELATIVE POSES (based on SVD) ------

    // SVD decompositon of E
    Matrix<double> U_E(3, 3, 0.0);
    Matrix<double> S_E(3, 3, 0.0);
    Matrix<double> V_E(3, 3, 0.0);
    svd_decompose(E, U_E, S_E, V_E);

    // construct W matrix
    mat3 W_(0, -1,  0,
            1,  0,  0,
            0,  0,  1);
    Matrix<double> W = to_Matrix(W_);

    // Calculate relative poses
    // R1 = (det U * W * Vt) * U * W * Vt
    // R2 = (det U * Wt * Vt) * U * Wt * Vt
    Matrix<double> R1_ = determinant(U_E * W * V_E.transpose()) * U_E * W * V_E.transpose();
    Matrix<double> R2_ = determinant(U_E * W.transpose() * V_E.transpose()) * U_E * W.transpose() * V_E.transpose();
    // t1,2 = +- U3
    Matrix<double> t1_(1, 3, U_E.get_column(2).data());
    Matrix<double> t2_ = t1_ * -1;

    std::cout << "\n"
        "Determinant R1: " << determinant(R1_) << " \n"
        "Determinant R2: " << determinant(R2_) << " \n";


    // STEP 2.3 - DETERMINE THE CORRECT RELATIVE POSE ---------------------

    // convert to <mat> and <vec>
    const mat3 R1 = to_mat3(R1_);
    const mat3 R2 = to_mat3(R2_);
    const vec3 t1 = to_vec3(t1_);
    const vec3 t2 = to_vec3(t2_);

    // find pair with highest number of points in front of both cameras
    int max = 0;
    int some_result;
    some_result = points_in_front(points_1, R1, t1, K_);     if (some_result > max) { R = R1; t = t1; max = some_result; }
    some_result = points_in_front(points_1, R1, t2, K_);     if (some_result > max) { R = R1; t = t2; max = some_result; }
    some_result = points_in_front(points_1, R2, t1, K_);     if (some_result > max) { R = R2; t = t1; max = some_result; }
    some_result = points_in_front(points_1, R2, t2, K_);     if (some_result > max) { R = R2; t = t2; max = some_result; }

    
    // determinant (R) = 1.0 (within a tiny threshold due to floating-point precision)
    // most (in theory it is 'all' but not in practice due to noise) estimated 3D points
    // are in front of the both cameras(i.e., z values w.r.t.camera is positive)

    std::cout << "\n"
        "R:\n " << R << " \n"
        "t:\n " << t << " \n\n";


    // ----------------- PART 3 --------------------------
    // --------- DETERMINE THE 3D COORDINATES ------------

    // STEP 3.0 - COMPUTE PROJECTION MATRIX FROM K, R and t

    mat34 M_(1.0f);
    mat34 M = K_ * M_;                      // M for first camera
    mat34 M_prime = get_M_matrix(R, t, K_); // M' for second camera


    // STEP 3.1 - COMPUTE THE 3D POINT USING THE LINEAR METHOD (SVD)

    vec4 M1 = M.row(0);
    vec4 M2 = M.row(1);
    vec4 M3 = M.row(2);

    vec4 M1_prime = M_prime.row(0);
    vec4 M2_prime = M_prime.row(1);
    vec4 M3_prime = M_prime.row(2);

    // vec3 x = get_3d_coordinates(M1, M2, M3, M1_prime, M2_prime, M3_prime, points_0[0], points_1[0]);

    // see function get_3d_coordinates
    // this function only works for 2 views 

    // OPTIONAL - NON-LINEAR LEAST-SQUARES REFINEMENT OF THE 3D POINT COMPUTED FROM THE LINEAR METHOD

    // STEP 3.2 - TRIANGULATE ALL CORRESPONDING IMAGE POINTS

    
    for (int i = 0; i < points_1.size(); ++i) {
        points_3d.push_back(get_3d_coordinates(M1, M2, M3, M1_prime, M2_prime, M3_prime, points_0[i], points_1[i]));
    }

    return points_3d.size() > 0;
}