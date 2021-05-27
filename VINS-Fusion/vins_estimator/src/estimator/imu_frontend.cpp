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
#include "imu_frontend.h"


gtsam::PreintegrationBase::Params ImuFrontend::convertVioImuParamsToGtsam(
    const ImuParams &imu_params) {
    gtsam::PreintegrationBase::Params preint_imu_params(imu_params.n_gravity_);
    preint_imu_params.gyroscopeCovariance =
        std::pow(imu_params.gyro_noise_density_, 2.0) *
        Eigen::Matrix3d::Identity();
    preint_imu_params.accelerometerCovariance =
        std::pow(imu_params.acc_noise_density_, 2.0) *
        Eigen::Matrix3d::Identity();
    preint_imu_params.integrationCovariance =
        std::pow(imu_params.imu_integration_sigma_, 2.0) *
        Eigen::Matrix3d::Identity();
    // TODO(Toni): REMOVE HARDCODED
    preint_imu_params.use2ndOrderCoriolis = false;
    return preint_imu_params;
}

boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
ImuFrontend::generateCombinedImuParams(const ImuParams& imu_params) {
    gtsam::PreintegrationParams gtsam_imu_params =
      ImuFrontend::convertVioImuParamsToGtsam(imu_params);
  boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
      combined_imu_params =
          boost::make_shared<gtsam::PreintegratedCombinedMeasurements::Params>(
              gtsam::PreintegratedCombinedMeasurements::Params(imu_params.n_gravity_));
  if (gtsam_imu_params.body_P_sensor) {
    combined_imu_params->setBodyPSensor(*gtsam_imu_params.getBodyPSensor());
  }
  if (gtsam_imu_params.omegaCoriolis) {
    combined_imu_params->setOmegaCoriolis(*gtsam_imu_params.getOmegaCoriolis());
  }
  combined_imu_params->setGyroscopeCovariance(
      gtsam_imu_params.getGyroscopeCovariance());
  combined_imu_params->setUse2ndOrderCoriolis(
      gtsam_imu_params.getUse2ndOrderCoriolis());
  combined_imu_params->setIntegrationCovariance(
      gtsam_imu_params.getIntegrationCovariance());
  combined_imu_params->setAccelerometerCovariance(
      gtsam_imu_params.getAccelerometerCovariance());
  ///< covariance of bias used for pre-integration
  // TODO(Toni): how come we are initializing like this?
  // We should parametrize perhaps this as well.
  combined_imu_params->biasAccOmegaInt = gtsam::I_6x6;
  ///< continuous-time "Covariance" describing
  ///< accelerometer bias random walk
  combined_imu_params->biasAccCovariance =
      std::pow(imu_params.acc_random_walk_, 2.0) * Eigen::Matrix3d::Identity();
  ///< continuous-time "Covariance" describing gyroscope bias random walk
  combined_imu_params->biasOmegaCovariance =
      std::pow(imu_params.gyro_random_walk_, 2.0) * Eigen::Matrix3d::Identity();
  return combined_imu_params;
}

boost::shared_ptr<gtsam::PreintegratedImuMeasurements::Params>
ImuFrontend::generateRegularImuParams(const ImuParams& imu_params) {
  boost::shared_ptr<gtsam::PreintegratedImuMeasurements::Params>
      regular_imu_params =
          boost::make_shared<gtsam::PreintegratedImuMeasurements::Params>(
              imu_params.n_gravity_);
  gtsam::PreintegrationParams gtsam_imu_params =
      ImuFrontend::convertVioImuParamsToGtsam(imu_params);
  if (gtsam_imu_params.body_P_sensor) {
    regular_imu_params->setBodyPSensor(*gtsam_imu_params.getBodyPSensor());
  }
  if (gtsam_imu_params.omegaCoriolis) {
    regular_imu_params->setOmegaCoriolis(*gtsam_imu_params.getOmegaCoriolis());
  }
  regular_imu_params->setGyroscopeCovariance(
      gtsam_imu_params.getGyroscopeCovariance());
  regular_imu_params->setUse2ndOrderCoriolis(
      gtsam_imu_params.getUse2ndOrderCoriolis());
  regular_imu_params->setIntegrationCovariance(
      gtsam_imu_params.getIntegrationCovariance());
  regular_imu_params->setAccelerometerCovariance(
      gtsam_imu_params.getAccelerometerCovariance());
  return regular_imu_params;
}

ImuFrontend::ImuFrontend(const ImuParams &imu_params, const ImuBias &imu_bias)
    : imu_params_(imu_params) {
    printf("imu_params (Iinit ImuFrontend) : \n");
    std::cout
        << "acc_noise_density_ : " << imu_params.acc_noise_density_ <<std::endl
        << "acc_random_walk_ : " << imu_params.acc_random_walk_ <<std::endl
        << "gyro_noise_density_ : " << imu_params.gyro_noise_density_ <<std::endl
        << "gyro_random_walk_ : " << imu_params.gyro_random_walk_ <<std::endl
        << "imu_integration_sigma_ : " << imu_params.imu_integration_sigma_ <<std::endl
        << "n_gravity_ : " << imu_params.n_gravity_ <<std::endl;
    printf("imu_bias (Iinit ImuFrontend) : \n");
    std::cout
        << "imu_bias : " << imu_bias <<std::endl;
    // pim_ = make_unique<gtsam::PreintegratedCombinedMeasurements>(generateCombinedImuParams(imu_params),
    //                                                              imu_bias);
    pim_ = make_unique<gtsam::PreintegratedImuMeasurements>(
          generateRegularImuParams(imu_params_), imu_bias);
    CHECK(pim_);
    {
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        latest_imu_bias_ = imu_bias;
    }
}

// ImuFrontend::CombinedPimPtr ImuFrontend::preintegrateImuMeasurements(
std::shared_ptr<gtsam::PreintegratedImuMeasurements> ImuFrontend::preintegrateImuMeasurements(
    const vector<pair<double, Eigen::Vector3d>> accVector,
    const vector<pair<double, Eigen::Vector3d>> gyrVector,
    const double current_time,
    const double previous_time) {
    CHECK(pim_) << "Pim not initialized.";
    for (size_t i = 0; i < accVector.size(); i++) {
        const gtsam::Vector3 &measured_acc = accVector[i].second;
        const gtsam::Vector3 &measured_omega = gyrVector[i].second;
        double dt;
        if (i == 0)
            dt = accVector[i].first - previous_time;
        else if (i == accVector.size() - 1)
            dt = current_time - accVector[i - 1].first;
        else
            dt = accVector[i].first - accVector[i - 1].first;
        // CHECK_GT(dt, 0.0) << "Imu dt is 0!";
        pim_->integrateMeasurement(measured_acc, measured_omega, dt);
    }
    // // why return make_unique?
    // return make_unique<gtsam::PreintegratedCombinedMeasurements>(
    //     dynamic_cast<const gtsam::PreintegratedCombinedMeasurements &>(*pim_));
    return make_unique<gtsam::PreintegratedImuMeasurements>(
        dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*pim_));
}
// ****************************ImuFrontend***********************************************//
