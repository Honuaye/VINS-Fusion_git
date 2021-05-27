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

#include "../featureTracker/feature_tracker.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../initial/solve_5pts.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include "feature_manager.h"
#include "parameters.h"

#include "gtsam-definitions.h"
#include "StereoCamera.h"

class BackendParams {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BackendParams() {}
    ~BackendParams() = default;
    //! Smart factor params
    gtsam::LinearizationMode linearizationMode_ = gtsam::HESSIAN;
    gtsam::DegeneracyMode degeneracyMode_ = gtsam::ZERO_ON_DEGENERACY; // 还没用起来??
    double smartNoiseSigma_ = 19.0;
    double rankTolerance_ = 1.0;
    //! max distance to triangulate point in meters
    double landmarkDistanceThreshold_ = 20.0;
    //! max acceptable reprojection error // before tuning: 3
    double outlierRejection_ = 8.0;
    double retriangulationThreshold_ = 1.0e-3;

    //! Between factor params
    bool addBetweenStereoFactors_ = true;
    // Inverse of variance
    double betweenRotationPrecision_ = 0.0;
    double betweenTranslationPrecision_ = 1 / (0.1 * 0.1);

    //! iSAM params
    double horizon_ = 6.0;
    // double horizon_ = 10.0;
    // double horizon_ = 3.0;
    int numOptimize_ = 2;
    bool useDogLeg_ = false;
    // bool useDogLeg_ = true;
    double wildfire_threshold_ = 0.001;
    double relinearizeThreshold_ = 1.0e-2;
    double relinearizeSkip_ = 1.0;

    //! No Motion params
    double noMotionPositionSigma_ = 1.0e-3;
    double noMotionRotationSigma_ = 1.0e-4;
    double zeroVelocitySigma_ = 1.0e-3;
    double constantVelSigma_ = 1.0e-2;
};

class FeatureTrack {
    // TODO(Toni): a feature track should have a landmark id...
    // TODO(Toni): a feature track should contain a pixel measurement per frame
    // but allow for multi-frame measurements at a time.
    // TODO(Toni): add getters for feature track length
 public:
    //! Observation: {FrameId, Px-Measurement}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::vector<std::pair<FrameId, gtsam::StereoPoint2>> obs_;

    // Is the lmk in the graph?
    bool in_ba_graph_ = false;

    FeatureTrack(FrameId frame_id, const gtsam::StereoPoint2 &px) {
        obs_.push_back(std::make_pair(frame_id, px));
    }
    void print() const {
        LOG(INFO) << "Feature track with cameras: ";
        for (size_t i = 0u; i < obs_.size(); i++) {
            std::cout << " " << obs_[i].first << " ";
        }
        std::cout << std::endl;
    }
};
using FeatureTracks = std::unordered_map<LandmarkId, FeatureTrack>;

class ImuFrontend {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using PimPtr = std::shared_ptr<gtsam::PreintegrationType>;
    using CombinedPimPtr =
        std::shared_ptr<gtsam::PreintegratedCombinedMeasurements>;
    using PimUniquePtr = std::unique_ptr<gtsam::PreintegrationType>;
    /* ------------------------------------------------------------------------
     * Class to do IMU preintegration.
     * [in] imu_params: IMU parameters used for the preintegration.
     * [in] imu_bias: IMU bias used to initialize PreintegratedImuMeasurements
     * !! The user of this class must update the bias and reset the integration
     * manually in order to preintegrate the IMU with the latest IMU bias coming
     * from the Backend optimization.
     */
    ImuFrontend() {}
    // Convert parameters for imu preintegration from the given ImuParams.
    static gtsam::PreintegrationType::Params convertVioImuParamsToGtsam(
        const ImuParams &imu_params);
    static boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params>
    generateCombinedImuParams(const ImuParams &imu_params);
    ImuFrontend(const ImuParams &imu_params, const ImuBias &imu_bias);
    ~ImuFrontend() = default;
    CombinedPimPtr preintegrateImuMeasurements(
        const vector<pair<double, Eigen::Vector3d>> accVector,
        const vector<pair<double, Eigen::Vector3d>> gyrVector,
        const double current_time,
        const double previous_time);

    // This should be called by the Backend, whenever there
    // is a new imu bias estimate. Note that we only store the new
    // bias, but we don't reset the pre-integration.This is because we
    // might have already started preintegrating measurements from
    // latest keyframe to the current frame using the previous bias,
    // which at the moment was the best last estimate.
    // The pre-integration is updated with the correct bias in the Backend.
    inline void updateBias(const ImuBias &imu_bias_prev_kf) {
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        latest_imu_bias_ = imu_bias_prev_kf;
    }

    // This should be called by the stereo Frontend, whenever there
    // is a new keyframe and we want to reset the integration to
    // use the latest imu bias.
    // THIS IS NOT THREAD-SAFE: pim_ is not protected.
    inline void resetIntegrationWithCachedBias(const ImuBias &imu_bias) {
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        latest_imu_bias_ = imu_bias;
        pim_->resetIntegrationAndSetBias(imu_bias);
    }

    // Reset gravity value in pre-integration.
    // This is needed for the online initialization.
    // THREAD-SAFE.
    inline void resetPreintegrationGravity(const gtsam::Vector3 &reset_value) {
        LOG(WARNING) << "Resetting value of gravity in ImuFrontend to: "
                     << reset_value;
        std::lock_guard<std::mutex> lock(imu_bias_mutex_);
        pim_->params()->n_gravity = reset_value;
        CHECK(gtsam::assert_equal(pim_->params()->getGravity(), reset_value));
        // TODO(Toni): should we update imu_params n_gravity for consistency?
        // imu_params_.n_gravity_ = reset_value;
    }
    inline gtsam::Vector3 getPreintegrationGravity() const {
        // TODO(Toni): why are we locking the imu_bias_mutex here???
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

class Estimator {
 public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Estimator();
    ~Estimator();
    void setParameter();
    void printData();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t,
                  const Vector3d &linearAcceleration,
                  const Vector3d &angularVelocity);
    void inputFeature(
        double t,
        const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
            &featureFrame);
    void inputImage(double t,
                    const cv::Mat &_img,
                    const cv::Mat &_img1 = cv::Mat());
    void processIMU(double t,
                    double dt,
                    const Vector3d &linear_acceleration,
                    const Vector3d &angular_velocity);

    void processImageGtsam(
        const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
        const double header,
        const std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> &pim);

    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0,
                        double t1,
                        vector<pair<double, Eigen::Vector3d>> &accVector,
                        vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri,
                             Vector3d &Pi,
                             Matrix3d &rici,
                             Vector3d &tici,
                             Matrix3d &Rj,
                             Vector3d &Pj,
                             Matrix3d &ricj,
                             Vector3d &ticj,
                             double depth,
                             Vector3d &uvi,
                             Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t,
                        Eigen::Vector3d linear_acceleration,
                        Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);

    enum SolverFlag { INITIAL, NON_LINEAR };
    enum MarginalizationFlag { MARGIN_OLD = 0, MARGIN_SECOND_NEW = 1 };

    std::ofstream save_estimator_times_;
    std::ofstream save_gtsam_op_times_;
    std::ofstream save_total_factors_;
    std::ofstream save_total_nonnull_factors_;
    std::ofstream save_pose_;
    std::ofstream save_position_;
    std::ofstream save_acc_bias_;
    std::ofstream save_gyr_bias_;

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    queue<pair<double, Eigen::Vector3d>> accBuf;
    queue<pair<double, Eigen::Vector3d>> gyrBuf;
    queue<
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>>
        featureBuf;
    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    FeatureTracker featureTracker;

    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;
    Eigen::Matrix4d motion_model_;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    // vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0,
        latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag;
    bool initThreadFlag;

    // GTSAM
    // Data:
    // TODO grows unbounded currently, but it should be limited to time horizon.
    FeatureTracks feature_tracks_;
    BackendParams backend_params_;
    ImuParams imu_params_;
    std::unique_ptr<ImuFrontend> imu_frontend_ = nullptr;
    std::unique_ptr<ImuFrontend> keyframe_imu_ = nullptr;
    ImuFrontend::CombinedPimPtr estimator_pim_ = nullptr;
    ImuFrontend::CombinedPimPtr keyframe_pim_ = nullptr;
    std::unique_ptr<StereoCamera> stereo_camera_ = nullptr;
    std::unique_ptr<CameraParams> left_cam_params_ = nullptr;
    std::unique_ptr<CameraParams> right_cam_params_ = nullptr;
    Eigen::Matrix3d left_K_;
    Eigen::Matrix3d right_K_;

    cv::Mat stereo_F12_;
    cv::Mat stereo_E12_;
    cv::Mat cv_stereo_F12_;
    cv::Mat cv_stereo_E12_;
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
    cv::Mat ComputeF12(
        const Eigen::Matrix3d &R1w,
        const Eigen::Matrix3d &R2w,
        const Eigen::Vector3d &t1w,
        const Eigen::Vector3d &t2w,
        cv::Mat *E12);
    bool CheckDistEpipolarLine(const cv::Point2f &p1,
                                const cv::Point2f &p2,
                                const cv::Mat &F12,
                                const double &tol);
    bool CheckDistEpipolarLine2(const cv::Point2f &p1,
                                const cv::Point2f &p2,
                                const cv::Mat &E12,
                                const double &tol);

 protected:
    bool optimize(const Timestamp &timestamp_kf_sec,
                  const FrameId &cur_id,
                  const size_t &max_iterations,
                  const gtsam::FactorIndices &extra_factor_slots_to_delete =
                      gtsam::FactorIndices());
    /**
     * @brief updateSmoother
     * @param result
     * @param new_factors_tmp
     * @param new_values
     * @param timestamps
     * @param delete_slots
     * @return False if the update failed, true otw.
     */
    bool updateSmoother(
        Smoother::Result *result,
        const gtsam::NonlinearFactorGraph &new_factors_tmp =
            gtsam::NonlinearFactorGraph(),
        const gtsam::Values &new_values = gtsam::Values(),
        const std::map<Key, double> &timestamps =
            gtsam::FixedLagSmoother::KeyTimestampMap(),
        const gtsam::FactorIndices &delete_slots = gtsam::FactorIndices());


    bool initStateAndSetPriors(
        const VioNavStateTimestamped &vio_nav_state_initial_seed);
    void addInitialPriorFactors(const FrameId &frame_id);

    // Set parameters for all types of factors.
    void setFactorsParams(
        const BackendParams &vio_params,
        gtsam::SharedNoiseModel *smart_noise,
        gtsam::SmartStereoProjectionParams *smart_factors_params,
        gtsam::SharedNoiseModel *no_motion_prior_noise,
        gtsam::SharedNoiseModel *zero_velocity_prior_noise,
        gtsam::SharedNoiseModel *constant_velocity_prior_noise);

    void setIsam2Params(const BackendParams &vio_params,
                        gtsam::ISAM2Params *isam_param) {
        // iSAM2 SETTINGS
        if (vio_params.useDogLeg_) {
            gtsam::ISAM2DoglegParams dogleg_params;
            dogleg_params.wildfireThreshold = vio_params.wildfire_threshold_;
            // dogleg_params.adaptationMode;
            // dogleg_params.initialDelta;
            // dogleg_params.setVerbose(false); // only for debugging.
            isam_param->optimizationParams = dogleg_params;
        } else {
            gtsam::ISAM2GaussNewtonParams gauss_newton_params;
            gauss_newton_params.wildfireThreshold =
                vio_params.wildfire_threshold_;
            isam_param->optimizationParams = gauss_newton_params;
        }
        // This can improve performence if linearization is expensive,
        // but can hurt performence if linearization is very cleap due to look up additional keys
        isam_param->setCacheLinearizedFactors(true);
        // Only relinearize variables whose linear delta magnitude is greater than this threshold (default: 0.1).
        isam_param->relinearizeThreshold = vio_params.relinearizeThreshold_;
        // Only relinearize any variables every relinearizeSkip calls to ISAM2::update (default:10) ???
        isam_param->relinearizeSkip = vio_params.relinearizeSkip_;
        // 对于 IncreamentalFixFlagSmoother , it should be true
        isam_param->findUnusedFactorSlots = true;

        /** Check variables for relinearization in tree-order, stopping the check once
         * a variable does not need to be relinearized (default: false). This can
         * improve speed by only checking a small part of the top of the tree.
         * However, variables below the check cut-off can accumulate significant
         * deltas without triggering relinearization. This is particularly useful in
         * exploration scenarios where real-time performance is desired over
         * correctness. Use with caution.
         * ***/
        // isam_param->enablePartialRelinearizationCheck = true;
        isam_param->setEvaluateNonlinearError(false);  // only for debugging
        isam_param->enableDetailedResults = false;     // only for debugging.
        isam_param->factorization = gtsam::ISAM2Params::CHOLESKY;  // QR

            // ISAM2Params(OptimizationParams _optimizationParams = ISAM2GaussNewtonParams(),
            //             RelinearizationThreshold _relinearizeThreshold = 0.1,
            //             int _relinearizeSkip = 10, bool _enableRelinearization = true,
            //             bool _evaluateNonlinearError = false,
            //             Factorization _factorization = ISAM2Params::CHOLESKY,
            //             bool _cacheLinearizedFactors = true,
            //             const KeyFormatter& _keyFormatter =
            //                 DefaultKeyFormatter,  ///< see ISAM2::Params::keyFormatter,
            //             bool _enableDetailedResults = false)
            //     : optimizationParams(_optimizationParams),
            //         relinearizeThreshold(_relinearizeThreshold),
            //         relinearizeSkip(_relinearizeSkip),
            //         enableRelinearization(_enableRelinearization),
            //         evaluateNonlinearError(_evaluateNonlinearError),
            //         factorization(_factorization),
            //         cacheLinearizedFactors(_cacheLinearizedFactors),
            //         keyFormatter(_keyFormatter),
            //         enableDetailedResults(_enableDetailedResults),
            //         enablePartialRelinearizationCheck(false),
            //         findUnusedFactorSlots(false) {}
    }

    using StereoMeasurement = std::pair<LandmarkId, gtsam::StereoPoint2>;
    using StereoMeasurements = std::vector<StereoMeasurement>;
    bool addVisualInertialStateAndOptimize(
        const Timestamp &timestamp_kf_sec,
        const StereoMeasurements &status_smart_stereo_measurements_kf,
        // const gtsam::PreintegrationType &pim,
        const std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> &pim,
        boost::optional<gtsam::Pose3> stereo_ransac_body_pose = boost::none);

    /*** For addVisualInertialStateAndOptimize  ****/
    // Set initial guess at current state.
    void addImuValues(
        const FrameId &cur_id,
        const std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> &pim);
    // Add imu factors:
    void addImuFactor(
        const FrameId &from_id,
        const FrameId &to_id,
        const std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> &pim);
    void addBetweenFactor(const FrameId &from_id,
                          const FrameId &to_id,
                          const gtsam::Pose3 &from_id_POSE_to_id);

    // Store stereo frame info into landmarks table:
    // returns landmarks observed in current frame.
    void addStereoMeasurementsToFeatureTracks(
        const int &curr_kf_id,
        const StereoMeasurements &stereoMeasurements_kf,
        LandmarkIds *landmarks_kf);

    // Add no motion factors in case of low disparity.
    void addZeroVelocityPrior(const FrameId &frame_id);
    void addNoMotionFactor(const FrameId &from_id, const FrameId &to_id);

    // Uses landmark table to add factors in graph.
    void addLandmarksToGraph(const LandmarkIds &landmarks_kf);
    // Adds a landmark to the graph for the first time.
    void addLandmarkToGraph(const LandmarkId &lm_id, const FeatureTrack &lm);
    void updateLandmarkInGraph(
        const LandmarkId &lmk_id,
        const std::pair<FrameId, StereoPoint2> &new_measurement);
    /*** For addVisualInertialStateAndOptimize  ****/
    bool deleteLmkFromFeatureTracks(const LandmarkId &lmk_id);

    // Update states.
    void updateStates(const FrameId &cur_id);
    void updateNewSmartFactorsSlots(
        const std::vector<LandmarkId> &lmk_ids_of_new_smart_factors_tmp,
        SmartFactorMap *old_smart_factors);


    void printSmootherInfo(const gtsam::NonlinearFactorGraph& new_factors_tmp,
                            const gtsam::FactorIndices& delete_slots,
                            const std::string& message = "CATCHING EXCEPTION",
                            const bool& showDetails = false) const;
    /********* cleanCheiralityLmk *********/
    void cleanCheiralityLmk(
        const gtsam::Symbol& lmk_symbol,
        gtsam::NonlinearFactorGraph* new_factors_tmp_cheirality,
        gtsam::Values* new_values_cheirality,
        std::map<Key, double>* timestamps_cheirality,
        gtsam::FactorIndices* delete_slots_cheirality,
        const gtsam::NonlinearFactorGraph& graph,
        const gtsam::NonlinearFactorGraph& new_factors_tmp,
        const gtsam::Values& new_values,
        const std::map<Key, double>& timestamps,
        const gtsam::FactorIndices& delete_slots);
    void deleteAllFactorsWithKeyFromFactorGraph(
        const gtsam::Key& key,
        const gtsam::NonlinearFactorGraph& new_factors_tmp,
        gtsam::NonlinearFactorGraph* factor_graph_output);
    // Returns if the key in timestamps could be removed or not.
    bool deleteKeyFromValues(const gtsam::Key& key,
                            const gtsam::Values& values,
                            gtsam::Values* values_output);
    // Returns if the key in timestamps could be removed or not.
    bool deleteKeyFromTimestamps(const gtsam::Key& key,
                                const std::map<Key, double>& timestamps,
                                std::map<Key, double>* timestamps_output);
    // Find all slots of factors that have the given key in the list of keys.
    void findSlotsOfFactorsWithKey(
        const gtsam::Key& key,
        const gtsam::NonlinearFactorGraph& graph,
        std::vector<size_t>* slots_of_factors_with_key);
    /********* cleanCheiralityLmk *********/


    // State estimates.
    // TODO(Toni): bundle these in a VioNavStateTimestamped.
    bool keyframe_;
    Timestamp timestamp_lkf_;
    ImuBias imu_bias_lkf_;  //!< Most recent bias estimate..
    gtsam::Vector3
        W_Vel_B_lkf_;  //!< Velocity of body at k-1 in world coordinates
    gtsam::Pose3 W_Pose_B_lkf_;  //!< Body pose at at k-1 in world coordinates.
    ImuBias imu_bias_prev_kf_;   //!< bias estimate at previous keyframe

    // State covariance. (initialize to zero)
    gtsam::Matrix state_covariance_lkf_ = Eigen::MatrixXd::Zero(15, 15);

    // Vision params.
    gtsam::SmartStereoProjectionParams smart_factors_params_;
    gtsam::SharedNoiseModel smart_noise_;
    // Pose of the left camera wrt body
    gtsam::Pose3 B_Pose_leftCam_;
    gtsam::Pose3 B_Pose_rightCam_;
    // Stores calibration, baseline.
    gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_ = nullptr;

    // State.
    //!< current state of the system.
    gtsam::Values state_;

    // ISAM2 smoother
    std::unique_ptr<Smoother> smoother_;

    // Values
    //!< new states to be added
    gtsam::Values new_values_;

    // Factors.
    //!< New factors to be added
    gtsam::NonlinearFactorGraph new_imu_prior_and_other_factors_;
    //!< landmarkId -> {SmartFactorPtr}
    LandmarkIdSmartFactorMap new_smart_factors_;
    //!< landmarkId -> {SmartFactorPtr, SlotIndex}
    SmartFactorMap old_smart_factors_;
    std::map<LandmarkId, bool> feature_flag_;
    // if SlotIndex is -1, means that the factor has not been inserted yet in
    // the graph

    // Counters.
    //! Last keyframe id.
    int last_kf_id_;
    //! Current keyframe id.
    int curr_kf_id_;

    std::map<int, int64_t> id_time_map_;
    std::map<int64_t, int> time_id_map_;

 private:
    //! No motion factors settings.
    gtsam::SharedNoiseModel zero_velocity_prior_noise_;
    gtsam::SharedNoiseModel no_motion_prior_noise_;
    gtsam::SharedNoiseModel constant_velocity_prior_noise_;

    //! Landmark count.
    int landmark_count_;

    //! Number of Cheirality exceptions
    size_t counter_of_exceptions_ = 0;
};
