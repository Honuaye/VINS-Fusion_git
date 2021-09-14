/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo {
    ResidualBlockInfo(ceres::CostFunction *_cost_function,
                      ceres::LossFunction *_loss_function,
                      std::vector<double *> _parameter_blocks,
                      std::vector<int> _drop_set)
        : cost_function(_cost_function),
          loss_function(_loss_function),
          parameter_blocks(_parameter_blocks),
          drop_set(_drop_set) {}
    void Evaluate();
    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;
    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        jacobians;
    Eigen::VectorXd residuals;
    int localSize(int size) { return size == 7 ? 6 : size; }
};

struct ThreadsStruct {
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size;  // global size
    std::unordered_map<long, int> parameter_block_idx;   // local size
};

class MarginalizationInfo {
 public:
    MarginalizationInfo() { valid_ = true; };
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    // 加残差块相关信息(优化变量、待marg的变量)
    void preMarginalize();
    // pos 为所有变量维度，m为需要marg掉的变量，n为需要保留的变量
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);
    //所有观测项
    std::vector<ResidualBlockInfo *> factors;
    // m : 为要marg掉的变量个数，也就是parameter_block_idx的总localSize，以double为单位，VBias为9，PQ为6
	// n : 为要保留下的优化变量的变量个数，n=localSize(parameter_block_size) – m
    int m, n;
    std::unordered_map<long, int> parameter_block_size;  // global size
    int sum_block_size;
    // <待marg的优化变量内存地址，在 //parameter_block_size中的id,以double为单位>
    std::unordered_map<long, int> parameter_block_idx;  // local size
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size;  // global size
    std::vector<int> keep_block_idx;   // local size
    // 最原始的数据值(会被固定???)
    std::vector<double *> keep_block_data;

    // 相当于 J_0 ，根据舒尔布后计算出来的H_* SVD分解出来的雅克比矩阵
    // 在后续迭代优化中，对边缘化先验项的雅克比矩阵一直用这个雅克比矩阵 J_0没有更新(2021.915)
    // 有个疑问： 好像很多资料都说VINS没有使用FEJ算法，但是这里的雅克比是一直不变的，所有算是没使用FEJ算法吗？？？
        // FEJ 算法就是保证边缘化对应残差块的迭代优化过程中，每次线性化的点保持一致，根据 J =偏倒(e_prior/X_0); FEJ就是要即保证X_0每次不变
        // 而，每次的 e_prior 是变化的，推理出 J 应该是需要变化的才能保证每次线性化的点 X_0 一致吗? 至于每次迭代的 J_ 该如何变化，如何保证都是同一个线性化点呢????
        // 雅克比矩阵
    Eigen::MatrixXd linearized_jacobians;
    // 相当于 e0_, 舒尔补后根据b0_和J_0反推出来的
    // 后续迭代更新过程中，会根据e0_计算每次更新后的 e_prior 公式为: e_prior = e0_ + J_0 * delta_x_prior；
        // 对应代码是：     Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals +
        //                     marginalization_info->linearized_jacobians * dx;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
    bool valid;
};

class MarginalizationFactor : public ceres::CostFunction {
 public:
    MarginalizationFactor(MarginalizationInfo *_marginalization_info);
    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const;

    MarginalizationInfo *marginalization_info;
};
