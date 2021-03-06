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



/// Convert a 3 by 3 matrix of type 'Matrix<double>' to mat3
mat3 to_mat3(Matrix<double>& M) {
    mat3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            result(i, j) = M(i, j);
    }
    return result;
}


/// Convert a 1 by 3 matrix of type 'Matrix<double>' to vec3
vec3 to_vec3(Matrix<double>& M) {
    vec3 result = vec3(M(0, 0), M(0, 1), M(0, 2));
    return result;
}


/// Convert M of type 'matN' (N can be any positive integer) to type 'Matrix<double>'
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


// Normalize coordinates of a group of points
void normalize(std::vector<vec3>& points, mat3& ST) {
    
    // Find centroid coordinates
    int pt_count = points.size();
    float sum_x = 0, sum_y = 0;
    for (int i = 0; i < pt_count; ++i) {
        sum_x += points[i].x;
        sum_y += points[i].y;
    }
    vec3 centroid{ sum_x / pt_count, sum_y / pt_count, 1 };

    // Calculate avg. distance
    float sum_dist = 0;
    for (int i = 0; i < pt_count; i++) {
        sum_dist += (points[i] - centroid).length();
    }
    float avg_dist = sum_dist / pt_count;

    // Get scaling factor (dist * s = sqrt(2))
    float s = sqrt(2) / avg_dist;

    // Create ST matrix
    ST = mat3{
        s,      0,      -s * centroid.x,
        0,      s,      -s * centroid.y,
        0,      0,              1
    };

    // change input vector to its normalised version
    for (int i = 0; i < points.size(); i++) {
        points[i] = ST * points[i];
    }
}


// Generate F matrix from normalized point pairs
mat3 get_F_matrix(const std::vector<vec3>& points_0n, const std::vector<vec3>& points_1n) {
    
    // Generate W Matrix
    Matrix<double> W(points_0n.size(), 9, 0.0);
    for (int i = 0; i < points_0n.size(); ++i) {
        double x1 = points_0n[i].x;
        double x2 = points_1n[i].x;
        double y1 = points_0n[i].y;
        double y2 = points_1n[i].y;
        W.set_row( { x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1.0 }, i);
    }

    // Decompose W with SVD
    int m = W.rows();
    int n = W.cols();
    Matrix<double> U_W(m, m, 0.0);
    Matrix<double> S_W(m, n, 0.0);
    Matrix<double> V_W(n, n, 0.0);
    svd_decompose(W, U_W, S_W, V_W);

    // Initialise and fill F matrix (F is a 3x3 matrix)
    mat3 F;

    int k = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            F(i, j) = V_W(k, 8);
            k++;
        }
    }

    return F;
}


// Generate M  matrix from K matrix
mat34 get_M_matrix(const mat3& K) {
    mat34 M = K * mat34(1.0f);
    return M;
}

// Generate M prime matrix from R, t and K matrices
mat34 get_Rt_matrix(const mat3& R, const vec3& t) {
    mat34 Rt;
    Rt.set_col(0, R.col(0));
    Rt.set_col(1, R.col(1));
    Rt.set_col(2, R.col(2));
    Rt.set_col(3, t);
    return Rt;
}

// Generate M prime matrix from R, t and K matrices
mat34 get_M_prime_matrix(const mat3& R, const vec3& t, const mat3& K) {
    mat34 Rt = get_Rt_matrix(R, t);
    mat34 M = K * Rt;
    return M;
}


// Triangulate 3D coordinates using both cameras M matrices
vec3 triangulate(const mat34& M, const mat34& Mp, const vec3& p1, const vec3& p2) {
    
    vec4 M1 = M.row(0);
    vec4 M2 = M.row(1);
    vec4 M3 = M.row(2);
    vec4 M1p = Mp.row(0);
    vec4 M2p = Mp.row(1);
    vec4 M3p = Mp.row(2);

    mat4 A(0.0);
    A.set_row(0, p1.x * M3 - M1);
    A.set_row(1, p1.y * M3 - M2);
    A.set_row(2, p2.x * M3p - M1p);
    A.set_row(3, p2.y * M3p - M2p);

    Matrix<double>A_ = to_Matrix(A);
    Matrix<double> U_A(4, 4, 0.0);
    Matrix<double> S_A(4, 4, 0.0);
    Matrix<double> V_A(4, 4, 0.0);
    svd_decompose(A_, U_A, S_A, V_A);

    std::vector<double> point_coord_h = V_A.get_column(3);

    vec3 point_coord; 
    for (int i = 0; i < 3; i++) {
        point_coord[i] = point_coord_h[i] / point_coord_h[3];
    }
    
    return point_coord;
}


/// To determine how many points are in front of camera for given relative camera pose
int points_in_front(const std::vector<vec3>& points_0, const std::vector<vec3>& points_1,
    const mat3& R, const vec3& t, const mat3& K)
{
    int found = 0;
    const mat34 Rt = get_Rt_matrix(R, t);
    const mat34 M = get_M_matrix(K);
    const mat34 M_prime = get_M_prime_matrix(R, t, K);

    for (int i = 0; i < points_0.size(); i++) {
        const vec3& p1 = points_0[i];
        const vec3& p2 = points_1[i];
        const vec3 p3d = triangulate(M, M_prime, p1, p2);
        const vec4 p3d_h = vec4{ p3d.x, p3d.y, p3d.z, 1.0 };
       
        // Check for both cameras
        if (p3d.z > 0 && (Rt * p3d_h).z > 0) found++;
    }
    return found;
}



/**
 * Function for reconstructing 3D geometry from corresponding image points.
 * @return True on success, otherwise false.
 */
bool Triangulation::triangulation(
    float fx, float fy,                     /// input: the focal lengths (same for both cameras)
    float cx, float cy,                     /// input: the principal point (same for both cameras)
    const std::vector<vec3>& points_0,      /// input: image points (in homogenous coordinates) in the 1st image.
    const std::vector<vec3>& points_1,      /// input: image points (in homogenous coordinates) in the 2nd image.
    std::vector<vec3>& points_3d,           /// output: reconstructed 3D points
    mat3& R,                                /// output: recovered rotation of 2nd camera (used for updating the viewer and visual inspection)
    vec3& t                                 /// output: recovered translation of 2nd camera (used for updating the viewer and visual inspection)
)   const
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

    mat3 ST;                                // normalization matrix for first camera
    mat3 ST_prime;                          // normalization matrix for second camera
    std::vector<vec3> p_0n = points_0;      // normalized points for first camera
    std::vector<vec3> p_1n = points_1;      // normalized points for second camera
    normalize(p_0n, ST);
    normalize(p_1n, ST_prime);


    // STEP 1.1 - LINEAR SOLUTION USING SVD ---------------------

    mat3 F = get_F_matrix(p_0n, p_1n);


    // STEP 1.2 - CONSTRAINT ENFORCEMENT (Based on SVD, Find the closest rank-2 matrix)

    // decompose F with SVD
    Matrix<double> F_ = to_Matrix(F);
    Matrix<double> U_F(3, 3, 0.0);
    Matrix<double> S_F(3, 3, 0.0);
    Matrix<double> V_F(3, 3, 0.0);
    svd_decompose(F_, U_F, S_F, V_F);
    
    //recompose constrained F with rank(F)=2
    S_F(2, 2) = 0;
    F_ = U_F * S_F * transpose(V_F);
    F = to_mat3(F_);


    // STEP 1.3 - DENORMALIZATION -----------------------------------

    F = transpose(ST_prime) * F * ST;
    std::cout << "denormalised F matrix \n" << F << "\n";



    // ------------------------ PART 2 ------------------------------------
    // --------- RECOVER RELATIVE POSE (R and t) FROM MATRIX F ------------

    // STEP 2.0 - CONSTRUCT K/K` MATRIX -----------------------------------

    // K Matrix with camera intrisic params
    // float fx, float fy,                     /// input: the focal lengths (same for both cameras)
    // float cx, float cy,                     /// input: the principal point (same for both cameras)
    mat3 K (fx, 0,  cx,
            0,  fy, cy,
            0,  0,  1);

    // STEP 2.1 - CALCULATE E MATRIX --------------------------------------

    // Essential matrix
    // E = K' * F * K
    mat3 E = transpose(K) * F * K;
    

    // STEP 2.2 - FIND THE 4 CANDIDATE RELATIVE POSES (based on SVD) ------

    // SVD decompositon of E
    Matrix<double> E_ = to_Matrix(E);
    Matrix<double> U_E(3, 3, 0.0);
    Matrix<double> S_E(3, 3, 0.0);
    Matrix<double> V_E(3, 3, 0.0);
    svd_decompose(E_, U_E, S_E, V_E);

    // Construct W matrix
    mat3 W (0, -1,  0,
            1,  0,  0,
            0,  0,  1);
    Matrix<double> W_ = to_Matrix(W);

    // Calculate relative poses
    // R1 = (det U * W * Vt) * U * W * Vt
    // R2 = (det U * Wt * Vt) * U * Wt * Vt
    Matrix<double> R1_ = determinant(U_E * W_ * V_E.transpose()) * U_E * W_ * V_E.transpose();
    Matrix<double> R2_ = determinant(U_E * W_.transpose() * V_E.transpose()) * U_E * W_.transpose() * V_E.transpose();
    // t1,2 = +- U3
    Matrix<double> t1_(1, 3, U_E.get_column(2).data());
    Matrix<double> t2_ = t1_ * -1;

    std::cout << "\n"
        "Determinant R1: " << determinant(R1_) << " \n"
        "Determinant R2: " << determinant(R2_) << " \n";


    // STEP 2.3 - DETERMINE THE CORRECT RELATIVE POSE ---------------------

    // Convert to <mat> and <vec>
    const mat3 R1 = to_mat3(R1_);
    const mat3 R2 = to_mat3(R2_);
    const vec3 t1 = to_vec3(t1_);
    const vec3 t2 = to_vec3(t2_);

    // Find pair with highest number of points in front of both cameras
    int max = 0;
    int some_result;
    some_result = points_in_front(points_0, points_1, R1, t1, K);     if (some_result > max) { R = R1; t = t1; max = some_result; }
    some_result = points_in_front(points_0, points_1, R1, t2, K);     if (some_result > max) { R = R1; t = t2; max = some_result; }
    some_result = points_in_front(points_0, points_1, R2, t1, K);     if (some_result > max) { R = R2; t = t1; max = some_result; }
    some_result = points_in_front(points_0, points_1, R2, t2, K);     if (some_result > max) { R = R2; t = t2; max = some_result; }

    // determinant (R) = 1.0 (within a tiny threshold due to floating-point precision)
    // most (in theory it is 'all' but not in practice due to noise) estimated 3D points
    // are in front of the both cameras(i.e., z values w.r.t.camera is positive)

    std::cout << "\n"
        "R:\n " << R << " \n"
        "t:\n " << t << " \n\n";



    // ----------------- PART 3 --------------------------
    // --------- DETERMINE THE 3D COORDINATES ------------

    // STEP 3.0 - COMPUTE PROJECTION MATRIX FROM K, R and t

    const mat34 M = get_M_matrix(K);                      // M for first camera
    const mat34 M_prime = get_M_prime_matrix(R, t, K);    // M' for second camera


    // STEP 3.2 - COMPUTE THE 3D POINT USING THE LINEAR METHOD (SVD)

    for (int i = 0; i < points_0.size(); ++i) {
        points_3d.push_back(triangulate(M, M_prime, points_0[i], points_1[i]));
    }

    return points_3d.size() > 0;
}