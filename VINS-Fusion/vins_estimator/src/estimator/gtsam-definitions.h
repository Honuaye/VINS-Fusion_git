
#pragma once

#include <Eigen/Dense>

#include <glog/logging.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/ISAM2Params.h>
#include <gtsam/slam/SmartFactorParams.h>

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

// #include "ImuFrontend.h"

using Timestamp = std::int64_t;

// Inertial containers.
// using ImuStamp = Timestamp;
using ImuStampS = Eigen::Matrix<Timestamp, 1, Eigen::Dynamic>;
// First 3 elements correspond to acceleration data [m/s^2]
// while the 3 last correspond to angular velocities [rad/s].
using ImuAccGyr = Eigen::Matrix<double, 6, 1>;
using ImuAcc = Eigen::Matrix<double, 3, 1>;
using ImuGyr = Eigen::Matrix<double, 3, 1>;
using ImuAccGyrS = Eigen::Matrix<double, 6, Eigen::Dynamic>;
using ImuBias = gtsam::imuBias::ConstantBias;

// Backend
// Gtsam types. // TODO remove these!!
using gtsam::Cal3_S2;
using gtsam::Key;
using gtsam::Point2;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::StereoPoint2;
using StereoCalibPtr = gtsam::Cal3_S2Stereo::shared_ptr;

using SymbolChar = unsigned char;
static constexpr SymbolChar kPoseSymbolChar = 'x';
static constexpr SymbolChar kVelocitySymbolChar = 'v';
static constexpr SymbolChar kImuBiasSymbolChar = 'b';
static constexpr SymbolChar kLandmarkSymbolChar = 'l';

typedef gtsam::IncrementalFixedLagSmoother Smoother;

// Definitions relevant to frame types
using FrameId = std::uint64_t;  // Frame id is used as the index of gtsam symbol
                                // (not as a gtsam key).
using PlaneId = std::uint64_t;
using LandmarkId = long int;  // -1 for invalid landmarks. // int would be too
                              // small if it is 16 bits!
using LandmarkIds = std::vector<LandmarkId>;

// Backend types
using SmartStereoFactor = gtsam::SmartStereoProjectionPoseFactor;
using SmartFactorParams = gtsam::SmartStereoProjectionParams;
using LandmarkIdSmartFactorMap =
    std::unordered_map<LandmarkId, SmartStereoFactor::shared_ptr>;
using Slot = long int;
using SmartFactorMap =
    gtsam::FastMap<LandmarkId, std::pair<SmartStereoFactor::shared_ptr, Slot>>;

using Landmark = gtsam::Point3;
using Landmarks = std::vector<Landmark>;
using PointWithId = std::pair<LandmarkId, Landmark>;
using PointsWithId = std::vector<PointWithId>;
using PointsWithIdMap = std::unordered_map<LandmarkId, Landmark>;
// using LmkIdToLmkTypeMap = std::unordered_map<LandmarkId, LandmarkType>;

class VioNavState {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VioNavState() : pose_(), velocity_(gtsam::Vector3::Zero()), imu_bias_() {}
    ~VioNavState() = default;

    VioNavState(const gtsam::Pose3 &pose,
                const gtsam::Vector3 &velocity,
                const gtsam::imuBias::ConstantBias &imu_bias)
        : pose_(pose), velocity_(velocity), imu_bias_(imu_bias) {}

    VioNavState(const gtsam::NavState &nav_state,
                const gtsam::imuBias::ConstantBias &imu_bias)
        : pose_(nav_state.pose()),
          velocity_(nav_state.velocity()),
          imu_bias_(imu_bias) {}
    gtsam::Pose3 pose_;
    gtsam::Vector3 velocity_;
    gtsam::imuBias::ConstantBias imu_bias_;
};

class VioNavStateTimestamped : public VioNavState {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VioNavStateTimestamped(const Timestamp &timestamp,
                           const VioNavState &vio_nav_state)
        : VioNavState(vio_nav_state), timestamp_(timestamp) {}

    VioNavStateTimestamped(const Timestamp &timestamp,
                           const gtsam::Pose3 &pose,
                           const gtsam::Vector3 &velocity,
                           const gtsam::imuBias::ConstantBias &imu_bias)
        : VioNavState(pose, velocity, imu_bias), timestamp_(timestamp) {}
    Timestamp timestamp_;
};

struct ImuParams {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuParams() {}
    ~ImuParams() = default;

 public:
    //   ImuPreintegrationType imu_preintegration_type_ =
    //       ImuPreintegrationType::kPreintegratedCombinedMeasurements;
    // 常规 IMU 参数
    double gyro_noise_density_ = 0.0;
    double gyro_random_walk_ = 0.0;
    double acc_noise_density_ = 0.0;
    double acc_random_walk_ = 0.0;
    double imu_time_shift_ = 0.0;  // Defined as t_imu = t_cam + imu_shift
    gtsam::Vector3 n_gravity_ = gtsam::Vector3::Zero();
    // ？？？
    double nominal_sampling_time_s_ = 0.0;
    double imu_integration_sigma_ = 0.0;
};

// }  // namespace Estimator
