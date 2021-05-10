/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g) {
    Eigen::Matrix3d R0;
    // g : 平均加速度 a_avert
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    // R0 = ng1.inverse() * ng2; ????
    // R0 到这里应该表示测量的加速度均值对应的旋转四元数相对于 (0,0,1)旋转四元数之间的旋转
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    // 获取 yaw 的欧拉角中的 yaw 角, yaw 角是偏航角(绕Z轴)
    double yaw = Utility::R2ypr(R0).x();
    // 将 -yaw 对应的旋转矩阵乘以 R0, 相当于对R0的yaw角归零吗
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
