#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <queue>
#include <fstream>
#include <thread>
#include <math.h>
#include <unordered_map>
#include <opencv2/core/eigen.hpp>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>


#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include "feature_manager.h"
#include "parameters.h"

#include "gtsam-definitions.h"
#include "stereo_camera.h"

class ImuFrontend {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using PimPtr = std::shared_ptr<gtsam::PreintegrationType>;
    using CombinedPimPtr = std::shared_ptr<gtsam::PreintegratedCombinedMeasurements>;
    using PimUniquePtr = std::unique_ptr<gtsam::PreintegrationType>;
    ImuFrontend() {}
    static gtsam::PreintegrationType::Params convertVioImuParamsToGtsam(
        const ImuParams &imu_params);
    static boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> generateCombinedImuParams(const ImuParams &imu_params);
    static boost::shared_ptr<gtsam::PreintegratedImuMeasurements::Params> generateRegularImuParams(const ImuParams &imu_params);
    ImuFrontend(const ImuParams &imu_params, const ImuBias &imu_bias);
    ~ImuFrontend() = default;
    std::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegrateImuMeasurements(
        const vector<pair<double, Eigen::Vector3d>> accVector,
        const vector<pair<double, Eigen::Vector3d>> gyrVector,
        const double current_time,
        const double previous_time);
    inline void updateBias(const ImuBias &imu_bias_prev_kf) {
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        latest_imu_bias_ = imu_bias_prev_kf;
    }
    inline void resetIntegrationWithCachedBias(const ImuBias &imu_bias) {
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        latest_imu_bias_ = imu_bias;
        pim_->resetIntegrationAndSetBias(imu_bias);
    }
    inline void resetPreintegrationGravity(const gtsam::Vector3 &reset_value) {
        LOG(WARNING) << "Resetting value of gravity in ImuFrontend to: "
                     << reset_value;
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        pim_->params()->n_gravity = reset_value;
        CHECK(gtsam::assert_equal(pim_->params()->getGravity(), reset_value));
    }
    inline gtsam::Vector3 getPreintegrationGravity() const {
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        return imu_params_.n_gravity_;
    }
    inline gtsam::PreintegrationType::Params getGtsamImuParams() const {
        return *(pim_->params());
    }
 private:
    ImuParams imu_params_;
    PimUniquePtr pim_ = nullptr;
    ImuBias latest_imu_bias_;
    mutable std::mutex imu_bias_mutex_;
};
