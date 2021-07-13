#include "estimator.h"
#include "../utility/visualization.h"
#include <boost/foreach.hpp>
#include <typeinfo>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

cv::Mat Estimator::SkewSymmetricMatrix(const cv::Mat &v) {
  return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
          v.at<float>(2), 0, -v.at<float>(0), -v.at<float>(1), v.at<float>(0),
          0);
}

cv::Mat Estimator::ComputeF12(
    const Eigen::Matrix3d &Rw1,
    const Eigen::Matrix3d &Rw2,
    const Eigen::Vector3d &tw1,
    const Eigen::Vector3d &tw2,
    cv::Mat *E12) {
  Eigen:Matrix3d tmp_R12 = Rw1.transpose() * Rw2;
  Eigen::Vector3d tmp_t12 = Rw1.transpose() * (tw2 - tw1);
  cv::Mat R12 = Utility::EigenMatrix3d2Mat(tmp_R12);
  cv::Mat t12 = Utility::EigenVector3d2Mat(tmp_t12);
  cv::Mat t12x = SkewSymmetricMatrix(t12);
  *E12 = t12x * R12;
  cv::Mat K1 = cv::Mat::eye(3, 3, CV_32F);
  K1.at<float>(0, 0) = INTRINSICS[0][0];
  K1.at<float>(1, 1) = INTRINSICS[0][1];
  K1.at<float>(0, 2) = INTRINSICS[0][2];
  K1.at<float>(1, 2) = INTRINSICS[0][3];
  cv::Mat K2 = cv::Mat::eye(3, 3, CV_32F);
  K2.at<float>(0, 0) = INTRINSICS[1][0];
  K2.at<float>(1, 1) = INTRINSICS[1][1];
  K2.at<float>(0, 2) = INTRINSICS[1][2];
  K2.at<float>(1, 2) = INTRINSICS[1][3];
  return K1.t().inv() * (*E12) * K2.inv();
}

bool Estimator::CheckDistEpipolarLine2(const cv::Point2f &p1,
                                       const cv::Point2f &p2,
                                       const cv::Mat &E12,
                                       const double &tol) {
  // Epipolar line in second image l = x1'F12 = [a b c]
  const float a = p1.x * E12.at<float>(0, 0) +
                  p1.y * E12.at<float>(1, 0) + E12.at<float>(2, 0);
  const float b = p1.x * E12.at<float>(0, 1) +
                  p1.y * E12.at<float>(1, 1) + E12.at<float>(2, 1);
  const float c = p1.x * E12.at<float>(0, 2) +
                  p1.y * E12.at<float>(1, 2) + E12.at<float>(2, 2);
  const float num = a * p2.x + b * p2.y + c;
  const float den = a * a + b * b;
  if (den == 0) {
    cout<<"(E: den=0 ";
    return false;
  }
  // 点到线的距离 d = sqrt((Ax0+By0+C)^2/(A^2+B^2))
  const float dsqr = num * num / den;
  std::cout<<"(E:"<<dsqr<<" ";
  return dsqr < tol;
}

bool Estimator::CheckDistEpipolarLine(const cv::Point2f &p1,
                                       const cv::Point2f &p2,
                                       const cv::Mat &F12,
                                       const double &tol) {
  // Epipolar line in second image l = x1'F12 = [a b c]
  const float a = p1.x * F12.at<float>(0, 0) +
                  p1.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
  const float b = p1.x * F12.at<float>(0, 1) +
                  p1.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
  const float c = p1.x * F12.at<float>(0, 2) +
                  p1.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);
  const float num = a * p2.x + b * p2.y + c;
  const float den = a * a + b * b;
  if (den == 0) {
    cout<<"F: den=0) ";
    return false;
  }
  // 点到线的距离 d = sqrt((Ax0+By0+C)^2/(A^2+B^2))
  const float dsqr = num * num / den;
  std::cout<<"F:"<<dsqr<<") ";
  return dsqr < tol;
}

void Estimator::printData(int index) {
    int last_index;
    if(index < 0) {
        index = frame_count;
    } else if (index == 0) {
        last_index = 0;
    }
    last_index = index - 1;
    cout<<"ID: "<< curr_kf_id_<<";  R: "<<Utility::R2ypr(Rs[index]).transpose()<<endl;;
    cout<<"Ps : "<<Ps[index].transpose()<<";   Vs : "<<Vs[index].transpose()<<endl;
    cout<<"[Bas: "<<Bas[index].transpose()<< "] [Bgs: "<<Bgs[index].transpose()<<"]"<<endl;

    Eigen::Vector3d dp = Ps[index]-Ps[last_index];
    Eigen::Vector3d dv = Vs[index]-Vs[last_index];
    Eigen::Vector3d dq = Utility::R2ypr(Rs[index])-Utility::R2ypr(Rs[last_index]);
    cout
        <<"dp: "<< dp.transpose()
        <<" dv: "<< dv.transpose()
        <<" dq: "<< dq.transpose() <<endl;
}

Estimator::Estimator() : f_manager{Rs} {
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

void Estimator::setFactorsParams(
    const BackendParams &vio_params,
    gtsam::SharedNoiseModel *smart_noise,
    gtsam::SmartStereoProjectionParams *smart_factors_params,
    gtsam::SharedNoiseModel *no_motion_prior_noise,
    gtsam::SharedNoiseModel *zero_velocity_prior_noise,
    gtsam::SharedNoiseModel *constant_velocity_prior_noise) {
    CHECK_NOTNULL(smart_noise);
    CHECK_NOTNULL(smart_factors_params);
    CHECK_NOTNULL(no_motion_prior_noise);
    CHECK_NOTNULL(zero_velocity_prior_noise);
    CHECK_NOTNULL(constant_velocity_prior_noise);
    // set smart_noise
    auto smart_noise_sigma = vio_params.smartNoiseSigma_;
    *smart_noise = gtsam::noiseModel::Isotropic::Sigma(3, smart_noise_sigma);
    // set SmartStereoFactorsParams
    CHECK_NOTNULL(smart_factors_params);
    *smart_factors_params = gtsam::SmartStereoProjectionParams();
    smart_factors_params->setRankTolerance(vio_params.rankTolerance_);
    smart_factors_params->setLandmarkDistanceThreshold(
        vio_params.landmarkDistanceThreshold_);
    smart_factors_params->setRetriangulationThreshold(
        vio_params.retriangulationThreshold_);
    smart_factors_params->setDynamicOutlierRejectionThreshold(
        vio_params.outlierRejection_);
    //! EPI: If set to true, will refine triangulation using LM.
    // if set true , updateNewSmartFactorsSlots() will throw exception
    smart_factors_params->setEnableEPI(false);
    // smart_factors_params->setEnableEPI(true);
    smart_factors_params->setLinearizationMode(gtsam::HESSIAN);
    smart_factors_params->setDegeneracyMode(gtsam::ZERO_ON_DEGENERACY);
    smart_factors_params->throwCheirality = false;
    smart_factors_params->verboseCheirality = false;

    // set NoMotionFactorsParams
    CHECK_NOTNULL(no_motion_prior_noise);
    gtsam::Vector6 sigmas;
    sigmas.head<3>().setConstant(vio_params.noMotionRotationSigma_);
    sigmas.tail<3>().setConstant(vio_params.noMotionPositionSigma_);
    *no_motion_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);

    // Zero velocity factors settings
    *zero_velocity_prior_noise =
        gtsam::noiseModel::Isotropic::Sigma(3u, vio_params.zeroVelocitySigma_);
    // Constant velocity factors settings
    *constant_velocity_prior_noise =
        gtsam::noiseModel::Isotropic::Sigma(3u, vio_params.constantVelSigma_);
}

Estimator::~Estimator() {
    std::cout<<"~Estimator"<<std::endl;
    if(save_pose_.is_open() ||
        save_position_.is_open() ||
        save_total_factors_.is_open() ||
        save_total_nonnull_factors_.is_open() ||
        save_acc_bias_.is_open() ||
        save_gyr_bias_.is_open() ||
        save_gtsam_op_times_.is_open() ||
        save_estimator_times_.is_open()
        ) {
        save_pose_.close();
        save_position_.close();
        save_total_factors_.close();
        save_total_nonnull_factors_.close();
        save_acc_bias_.close();
        save_gyr_bias_.close();
        save_gtsam_op_times_.close();
        save_estimator_times_.close();
    }
    if (MULTIPLE_THREAD) {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState() {
    mProcess.lock();
    while (!accBuf.empty()) accBuf.pop();
    while (!gyrBuf.empty()) gyrBuf.pop();
    while (!featureBuf.empty()) featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;
    first_image_ = true;

    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr) {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false, sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr) delete tmp_pre_integration;
    tmp_pre_integration = nullptr;
    // last_marginalization_parameter_blocks.clear();
    f_manager.clearState();
    failure_occur = 0;
    mProcess.unlock();
}

void Estimator::setParameter() {
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl
             << ric[i] << endl
             << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);
    // /******* GET GTSAM PARAMS ********/
    stereo_E12_ = cv::Mat::eye(3, 3, CV_32F); 
    stereo_F12_ = ComputeF12(ric[0], ric[1], tic[0], tic[1], &stereo_E12_);
    left_K_.setZero();
    left_K_(0,0) = INTRINSICS[0][0]; left_K_(0,1) = 0; left_K_(0,2) = INTRINSICS[0][2];
    left_K_(1,0) = 0; left_K_(1,1) = INTRINSICS[0][1]; left_K_(1,2) = INTRINSICS[0][3];
    left_K_(2,0) = 0; left_K_(2,1) = 0; left_K_(2,2) = 1;
    right_K_.setZero();
    right_K_(0,0) = INTRINSICS[1][0]; right_K_(0,1) = 0; right_K_(0,2) = INTRINSICS[1][2];
    right_K_(1,0) = 0; right_K_(1,1) = INTRINSICS[1][1]; right_K_(1,2) = INTRINSICS[1][3];
    right_K_(2,0) = 0; right_K_(2,1) = 0; right_K_(2,2) = 1;

    cv::Size image_size(COL, ROW);
    Eigen::Matrix4d Tic0, Tic1;
    Tic0.block<3, 3>(0, 0) = RIC[0];
    Tic0.block<3, 1>(0, 3) = TIC[0];
    Tic1.block<3, 3>(0, 0) = RIC[1];
    Tic1.block<3, 1>(0, 3) = TIC[1];
        // INTRINSICS[index] = {m_fx, m_fy, m_cx, m_cy};
        // DISTORTION_COEFF[index] = {m_k1, m_k2, m_p1, m_p2};
    gtsam::Pose3 left_body_Pose_cam(Tic0);
    left_cam_params_ = std::make_unique<CameraParams>(
        image_size,
        left_body_Pose_cam,
        INTRINSICS[0],
        DISTORTION_COEFF[0]);
    gtsam::Pose3 right_body_Pose_cam(Tic1);
    right_cam_params_ = std::make_unique<CameraParams>(
        image_size,
        right_body_Pose_cam,
        INTRINSICS[1],
        DISTORTION_COEFF[1]);
    stereo_camera_ = std::make_unique<StereoCamera>(*left_cam_params_, *right_cam_params_);
    // Get left camera pose after rectification with respect to the body frame.
    B_Pose_leftCam_ = stereo_camera_->getBodyPoseLeftCamRect();
    B_Pose_rightCam_ = stereo_camera_->getBodyPoseRightCamRect();
    // stereo calibration_: stereo camera calibration after undistortion and rectification.
    stereo_calibration_ = stereo_camera_->getStereoCalib();
    cv_stereo_E12_ = cv::Mat::eye(3, 3, CV_32F);
    Eigen::Vector3d t1 = B_Pose_leftCam_.translation();
    Eigen::Vector3d t2 = B_Pose_rightCam_.translation();
    Eigen::Matrix3d r1 = B_Pose_leftCam_.rotation().matrix();
    Eigen::Matrix3d r2 = B_Pose_rightCam_.rotation().matrix();
    cv_stereo_F12_ = ComputeF12(r1, r2, t1, t2, &cv_stereo_E12_);

    printf("EF: ---------------------\n");
    cout<<"stereo_E12_: "<<endl<<stereo_E12_<<endl;
    cout<<"stereo_F12_: "<<endl<<stereo_F12_<<endl;
    cout<<"cv_stereo_E12_: "<<endl<<cv_stereo_E12_<<endl;
    cout<<"cv_stereo_F12_: "<<endl<<cv_stereo_F12_<<endl;
    cout<<"r1: "<<r1<<endl;
    cout<<"r2: "<<r2<<endl;
    cout<<"t1: "<<t1.transpose()<<endl;
    cout<<"t2: "<<t2.transpose()<<endl;
    B_Pose_leftCam_.print("B_Pose_leftCam_");
    B_Pose_rightCam_.print("B_Pose_rightCam_");
    stereo_calibration_->print("stereo_calibration_");

    cout<<"RIC[0]= "<<Utility::R2ypr(RIC[0]).transpose()<<endl;
    cout<<"RIC[1]= "<<Utility::R2ypr(RIC[1]).transpose()<<endl;
    cout<<"r_s= "<<Utility::R2ypr(r1).transpose()<<endl;
    // K:
        // 434.656 0 357.564
        // 0 434.656 255.396
        // 0 0 1
    // Baseline: 0.110078
    /********* init GTSAM ************/
    imu_params_.gyro_noise_density_ = GYR_N;
    imu_params_.acc_noise_density_ = ACC_N;
    imu_params_.gyro_random_walk_ = GYR_W;
    imu_params_.acc_random_walk_ = ACC_W;
    imu_params_.n_gravity_ = gtsam::Vector3(0.0, 0.0, G.z());
    // what's this ？？
    imu_params_.imu_integration_sigma_ = 1.0;
    imu_params_.nominal_sampling_time_s_ = 200.0;
    // Values
    W_Vel_B_lkf_ = gtsam::Vector3::Zero();
    W_Pose_B_lkf_ = gtsam::Pose3::identity();
    imu_bias_lkf_ = ImuBias();
    imu_bias_prev_kf_ = ImuBias();
    // index
    landmark_count_ = 0;
    timestamp_lkf_ = -1;
    last_kf_id_ = -1;
    curr_kf_id_ = 0;
    // Init imu frontend
    keyframe_imu_ = std::make_unique<ImuFrontend>(imu_params_, imu_bias_lkf_);
    // Init smoother.
    gtsam::ISAM2Params isam_param;
    setIsam2Params(backend_params_, &isam_param);
    // backend_params_.horizon_ : sliding_window size
    smoother_ = make_unique<Smoother>(backend_params_.horizon_, isam_param);
    // Set parameters for all factors.
    setFactorsParams(backend_params_, &smart_noise_, &smart_factors_params_,
                     &no_motion_prior_noise_, &zero_velocity_prior_noise_,
                     &constant_velocity_prior_noise_);
    // /******* GET GTSAM PARAMS ********/
    {
        // get gt
        std::string ground_truth_path = GROUND_TRUTH_PATH + "data.csv";
        std::ifstream gt_quaternion;
        gt_quaternion.open(ground_truth_path, std::ios::in);
        if(gt_quaternion.is_open()) {
            int size = 0;
            std::string gt;
            while (getline(gt_quaternion, gt)) {
                if(size == 0){
                    size++;
                    continue;
                }
                vector<double> data_vec;
                std::string split_flag(",");
                while (gt.find_first_of(split_flag) != std::string::npos) {
                    int end = gt.find_first_of(split_flag);
                    string str = gt.substr(0, end);
                    gt = gt.substr(end + 1);
                    data_vec.push_back(atof(str.c_str()));
                }
                data_vec.push_back(atof(gt.c_str()));
                double time = data_vec[0] * 1e-9;
                Eigen::Vector3d tmp_p(data_vec[1], data_vec[2], data_vec[3]);
                Quaterniond tmp_q(data_vec[4], data_vec[5], data_vec[6], data_vec[7]);
                Eigen::Vector3d tmp_v(data_vec[8], data_vec[9], data_vec[10]);
                Eigen::Vector3d tmp_bg(data_vec[11], data_vec[12], data_vec[13]);
                Eigen::Vector3d tmp_ba(data_vec[14], data_vec[15], data_vec[16]);
                gt_p_.insert(make_pair(time, tmp_p));
                gt_q_.insert(make_pair(time, tmp_q));
                gt_v_.insert(make_pair(time, tmp_v));
                gt_bg_.insert(make_pair(time, tmp_bg));
                gt_ba_.insert(make_pair(time, tmp_ba));
                size++;
            }
        }
        for(auto it : gt_ba_){
            cout<<it.first<<", "<<it.second.transpose()<<endl;
        }
        // set save
        std::string horizon_string = std::to_string(u_int(backend_params_.horizon_));
        std::string save_path = MY_OUTPUT_FOLDER + "gtsam_h"+horizon_string+"_";
        std::string save_pose_path = save_path + std::string("pose.txt");
        std::string save_position_path = save_path + "position.txt";
        std::string save_acc_bias_path = save_path + "accBias.txt";
        std::string save_gyr_bias_path = save_path + "gyrBias.txt";
        std::string total_factors = save_path + "total_factors.txt";
        std::string total_non_factors = save_path + "total_non_factors.txt";
        std::string save_gtsam_op_time_path = save_path + "gtsam_op_time.txt";
        std::string save_estimator_time_path = save_path + "estimator_time.txt";
        save_pose_.open(save_pose_path, std::ios::out | std::ios::trunc);
        save_position_.open(save_position_path, std::ios::out | std::ios::trunc);
        save_acc_bias_.open(save_acc_bias_path, std::ios::out | std::ios::trunc);
        save_gyr_bias_.open(save_gyr_bias_path, std::ios::out | std::ios::trunc);
        save_total_factors_.open(total_factors, std::ios::out | std::ios::trunc);
        save_total_nonnull_factors_.open(total_non_factors, std::ios::out | std::ios::trunc);
        save_gtsam_op_times_.open(save_gtsam_op_time_path, std::ios::out | std::ios::trunc);
        save_estimator_times_.open(save_estimator_time_path, std::ios::out | std::ios::trunc);
        save_pose_.clear();
        save_position_.clear();
        save_acc_bias_.clear();
        save_gyr_bias_.clear();
        save_total_factors_.clear();
        save_total_nonnull_factors_.clear();
        save_gtsam_op_times_.clear();
        save_estimator_times_.clear();
        if(!save_pose_.is_open() ||
            !save_position_.is_open() ||
            !save_acc_bias_.is_open() ||
            !save_total_factors_.is_open() ||
            !save_total_nonnull_factors_.is_open() ||
            !save_gtsam_op_times_.is_open() ||
            !save_estimator_times_.is_open() ||
            !save_gyr_bias_.is_open()) {
            std::cout<<"Failed to open save file !!!"<<std::endl;
            assert(0);
        }
    }
    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag) {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::inputImage(double t,
                           const cv::Mat &_img,
                           const cv::Mat &_img1) {
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;
    if (_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img);
    else
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    if (SHOW_TRACK) {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }
    if (MULTIPLE_THREAD) {
        if (inputImageCnt % 2 == 0) {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    } else {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
}

void Estimator::inputIMU(double t,
                         const Vector3d &linearAcceleration,
                         const Vector3d &angularVelocity) {
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    mBuf.unlock();
    if (solver_flag == NON_LINEAR) {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(
    double t,
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
        &featureFrame) {
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if (!MULTIPLE_THREAD) processMeasurements();
}

bool Estimator::getIMUInterval(
    double t0,
    double t1,
    vector<pair<double, Eigen::Vector3d>> &accVector,
    vector<pair<double, Eigen::Vector3d>> &gyrVector) {
    if (accBuf.empty()) {
        printf("not receive imu\n");
        return false;
    }
    if (t1 <= accBuf.back().first) {
        while (accBuf.front().first <= t0) {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1) {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    } else {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t) {
    if (!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements() {
    while (1) {
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if (!featureBuf.empty()) {
            static int process_num=0;
            TicToc estimator_time;
            feature = featureBuf.front();
            curTime = feature.first + td;
            if(first_image_) {
                prevTime = curTime;
                first_image_ = false;
            }
            while (1) {
                if ((!USE_IMU || IMUAvailable(feature.first + td)))
                    break;
                else {
                    printf("wait for imu ... \n");
                    if (!MULTIPLE_THREAD) return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if (USE_IMU) {
                getIMUInterval(prevTime, curTime, accVector, gyrVector);
            }
            featureBuf.pop();
            mBuf.unlock();
            Eigen::Matrix4d old_T;
            Eigen::Matrix4d predict_cur_T;
            getPoseInWorldFrame(old_T);
            predict_cur_T = old_T * motion_model_;
            if (USE_IMU) {
                if (!initFirstPoseFlag) initFirstIMUPose(accVector);
                Eigen::Vector3d tmp(Ps[WINDOW_SIZE]);
                for (size_t i = 0; i < accVector.size(); i++) {
                    double dt;
                    if (i == 0) {
                        dt = accVector[i].first - prevTime;
                    } else if (i == accVector.size() - 1) {
                        dt = curTime - accVector[i - 1].first;
                    } else {
                        dt = accVector[i].first - accVector[i - 1].first;
                    }
                    processIMU(accVector[i].first, dt, accVector[i].second,
                               gyrVector[i].second);
                }
                //****GTSAM****//
                keyframe_pim_ = keyframe_imu_->preintegrateImuMeasurements(
                        accVector, gyrVector, curTime, prevTime);
                auto ypr = Utility::R2ypr(pre_integrations[frame_count]->delta_q.toRotationMatrix());
                if (solver_flag != INITIAL) {
                    Ps[WINDOW_SIZE] = tmp;
                }
                //****GTSAM****//
            }
            mProcess.lock();
            CHECK(keyframe_pim_);
            processImageGtsam(feature.second, feature.first, keyframe_pim_);
            prevTime = curTime;
            printStatistics(*this, 0);
            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);
            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            // // pubPointCloud(*this, header);
            if(lmk_ids_to_3d_points_in_time_horizon_.size() > 0) {
                pubGtsamPointCloud(*this, header);
            }
            pubKeyframe(*this);
            pubTF(*this, header);
            if(solver_flag == NON_LINEAR)
            {
                {
                    Eigen::Matrix4d Tw0;
                    Eigen::Matrix3d R0 = gt_q_.begin()->second.toRotationMatrix();
                    Eigen::Vector3d t0 = gt_p_.begin()->second;
                    if(save_first_data_) {
                        Tw0.setIdentity();
                        Tw0.block<3, 3>(0, 0) = R0;
                        Tw0.block<3, 1>(0, 3) = t0;
                    }
                    while (gt_p_.begin()->first < curTime) {
                        if(save_first_data_) {
                            gap_q_ = gt_q_.begin()->second;
                            gap_p_ = gt_p_.begin()->second;
                            gap_v_ = gt_v_.begin()->second;
                            gap_bg_ = gt_bg_.begin()->second;
                            gap_ba_ = gt_ba_.begin()->second;
                        }
                        gt_p_.erase(gt_p_.begin());
                        gt_q_.erase(gt_q_.begin());
                        gt_v_.erase(gt_v_.begin());
                        gt_bg_.erase(gt_bg_.begin());
                        gt_ba_.erase(gt_ba_.begin());
                    }
                    cout<<setprecision(16)<<"gt_p_: "<<gt_p_.begin()->first<<"; urTime:"<<curTime<<endl;
                    if(save_first_data_) {
                        T01_.setIdentity();
                        R01_ = gt_q_.begin()->second.toRotationMatrix();
                        t01_ = gt_p_.begin()->second;
                        T01_.block<3, 3>(0, 0) = R01_;
                        T01_.block<3, 1>(0, 3) = t01_;
                        T01_ = Tw0.transpose() * T01_;
                        R01_ = T01_.block<3, 3>(0, 0);
                        t01_ = T01_.block<3, 1>(0, 3);
                        cout<<"gap_p_: "<<gap_p_.transpose()<<endl;
                        save_pose_ << "q(w, x, y, z); gt_q(); ypr; init rotation gt: "
                            << gt_q_.begin()->second.w()
                            << gt_q_.begin()->second.x()
                            << gt_q_.begin()->second.y()
                            << gt_q_.begin()->second.z()
                            <<endl;
                        save_position_ << "xyz, gt_xyz; init position gt: " << gt_p_.begin()->second.transpose()<<endl;
                        save_acc_bias_ << "acc_bias(xyz), gt_acc_bias" <<endl;
                        save_gyr_bias_ << "gyr_bias(xyz), gt_gyr_bias" <<endl;
                        save_first_data_ = false;
                    }
                }
                // Eigen::Matrix3d R_save = R01_* Rs[frame_count];
                // Eigen::Vector3d t_save = R01_ * Ps[frame_count] + t01_;
                Eigen::Matrix3d R_save = R01_* Rs[frame_count];
                Eigen::Vector3d t_save = R01_ * Ps[frame_count] + t01_;
                std::string current_time = std::to_string(curTime);
                save_estimator_times_ << current_time <<","<< estimator_time.toc()<<endl;
                Eigen::Vector3d ypr =  Utility::R2ypr(R_save);
                Quaterniond q{R_save};
                save_pose_
                    << current_time<<","
                    <<q.w()<<","
                    <<q.x()<<","
                    <<q.y()<<","
                    <<q.z()<<","
                    <<gt_q_.begin()->second.w()<<","
                    <<gt_q_.begin()->second.x()<<","
                    <<gt_q_.begin()->second.y()<<","
                    <<gt_q_.begin()->second.z()<<","
                    <<ypr(0)<<","
                    <<ypr(1)<<","
                    <<ypr(2)
                    << std::endl;
                save_position_
                    << current_time<<","
                    <<t_save(0)<<","
                    <<t_save(1)<<","
                    <<t_save(2)<<","
                    <<gt_p_.begin()->second(0)<<","
                    <<gt_p_.begin()->second(1)<<","
                    <<gt_p_.begin()->second(2)
                    << std::endl
                    ;
                save_acc_bias_
                    << current_time<<","
                    <<Bas[frame_count](0)<<","
                    <<Bas[frame_count](1)<<","
                    <<Bas[frame_count](2)<<","
                    <<gt_ba_.begin()->second(0)<<","
                    <<gt_ba_.begin()->second(1)<<","
                    <<gt_ba_.begin()->second(2)
                    << std::endl
                    ;
                save_gyr_bias_
                    << current_time<<","
                    <<Bgs[frame_count](0)<<","
                    <<Bgs[frame_count](1)<<","
                    <<Bgs[frame_count](2)<<","
                    <<gt_bg_.begin()->second(0)<<","
                    <<gt_bg_.begin()->second(1)<<","
                    <<gt_bg_.begin()->second(2)
                    << std::endl
                    ;
            }
            mProcess.unlock();
        }
        if (!MULTIPLE_THREAD) break;
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }

}

void Estimator::initFirstIMUPose(
    vector<pair<double, Eigen::Vector3d>> &accVector) {
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    // return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for (size_t i = 0; i < accVector.size(); i++) {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r) {
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

void Estimator::processIMU(double t,
                           double dt,
                           const Vector3d &linear_acceleration,
                           const Vector3d &angular_velocity) {
    if (!first_imu) {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    if (!pre_integrations[frame_count]) {
        pre_integrations[frame_count] = new IntegrationBase{
            acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0) {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration,
                                                 angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration,
                                       angular_velocity);
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        // Rs : Rbw
        // accb = Rbw * accw
        // linear_acceleration 是测量到的世界坐标系下的东西
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

bool Estimator::failureDetection() {
    if (Bas[WINDOW_SIZE].norm() > 2.5) {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        // return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0) {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5) {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 2) {
        ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50) {
        // ROS_INFO(" big delta_angle ");
        // static int num = 0;
        // num++;
        // if(num > 4) {
        //     return true;
        // }
    }
    return false;
}

void Estimator::slideWindow() {
    TicToc t_margin;
    double t_0 = Headers[0];
    back_R0 = Rs[0];
    back_P0 = Ps[0];
    if (frame_count == WINDOW_SIZE) {
        for (int i = 0; i < WINDOW_SIZE; i++) {
            Headers[i] = Headers[i + 1];
            Rs[i].swap(Rs[i + 1]);
            Ps[i].swap(Ps[i + 1]);
            if (USE_IMU) {
                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
        }
        Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
        Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
        Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

        if (USE_IMU) {
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{
                acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
        }
        if (true || solver_flag == INITIAL) {
            map<double, ImageFrame>::iterator it_0;
            it_0 = all_image_frame.find(t_0);
            if(it_0->second.pre_integration) {
                delete it_0->second.pre_integration;
            }
            all_image_frame.erase(all_image_frame.begin(), it_0);
        }
        slideWindowOld();
    }
}

void Estimator::slideWindowOld() {
    sum_of_back++;
    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if(shift_depth) {
        return;
    }
    f_manager.removeBack();
}

void Estimator::slideWindowNew() {
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame() {
    if (frame_count < 2) return;
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature) {
        if (it_per_id.estimated_depth > 0) {
            int firstIndex = it_per_id.start_frame;
            int lastIndex =
                it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            // printf("cur frame index  %d last frame index %d\n", frame_count,
            // lastIndex);
            if ((int)it_per_id.feature_per_frame.size() >= 2 &&
                lastIndex == frame_count) {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j =
                    ric[0] * (depth * it_per_id.feature_per_frame[0].point) +
                    tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() *
                                     (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    // printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri,
                                    Vector3d &Pi,
                                    Matrix3d &rici,
                                    Vector3d &tici,
                                    Matrix3d &Rj,
                                    Vector3d &Pj,
                                    Matrix3d &ricj,
                                    Vector3d &ticj,
                                    double depth,
                                    Vector3d &uvi,
                                    Vector3d &uvj) {
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::fastPredictIMU(double t,
                               Eigen::Vector3d linear_acceleration,
                               Eigen::Vector3d angular_velocity) {
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr =
        0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates() {
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while (!tmp_accBuf.empty()) {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}

// // GTSAM
void Estimator::processImageGtsam(
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    const double header,
    const std::shared_ptr<gtsam::PreintegratedImuMeasurements> &pim) {
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
        keyframe_ = true;
    } else {
        keyframe_ = false;
    }
    Headers[frame_count] = header;
    timestamp_lkf_ = header;
    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{
        acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    if (solver_flag == INITIAL) {
        if (STEREO && USE_IMU) {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE) {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin();
                     frame_it != all_image_frame.end(); frame_it++) {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++) {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                Eigen::Matrix4d curT;
                getPoseInWorldFrame(curT);
                // gtsam::Pose3 initial_pose_guess(curT);
                // gtsam::Vector3 velocity_guess = gtsam::Vector3(Vs[frame_count]);
                gtsam::Pose3 initial_pose_guess;
                gtsam::Vector3 velocity_guess;
                printData();
                // 150:
                    // gyroscope bias initial calibration -0.0025655  0.0285788  0.0844428
                    // VINS master: gyroscope bias initial calibration -0.00257274    0.020599   0.0823816
                // 400 : 
                    // -0.00180214   0.0290869   0.0838947
                    // -0.00180214   0.0290869   0.0838947  (需要跟提取特征点阈值有关)
                    // gyroscope bias initial calibration -0.00180214   0.0290869   0.0838947  
                ImuBias imu_bias_guess(Bas[frame_count], Bgs[frame_count]);
                keyframe_imu_->resetIntegrationWithCachedBias(imu_bias_guess);
                VioNavState initial_state_estimate =
                    VioNavState(initial_pose_guess, velocity_guess, imu_bias_guess);
                if (initStateAndSetPriors(
                    VioNavStateTimestamped((int64_t)curTime, initial_state_estimate))) {
                    solver_flag = NON_LINEAR;
                    printf("\n Finish init  -------------------------------\n");
                } else {
                    printf("\n init State And SetPriors fail! Try again");
                }
                slideWindow();
            }
        }
        if (frame_count < WINDOW_SIZE) {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
    }
    else
    {
        last_kf_id_ = curr_kf_id_;
        ++curr_kf_id_;
        TicToc t_solve;
        bool is_smoother_ok;
        StereoMeasurements smart_stereo_measurements;
        smart_stereo_measurements.reserve(image.size());
        std::vector<cv::Point2f> left_pts, right_pts;
        std::vector<cv::Point2f> left_pts_rectified, right_pts_rectified;
        int good = 0;
        int average = 0;
        int bad = 0;
        int total = 0;
        for (auto &id_pts : image) {
            LandmarkId landmark_id = id_pts.first;
            double u_l = std::numeric_limits<double>::quiet_NaN();
            double v_l = std::numeric_limits<double>::quiet_NaN();
            double u_r = std::numeric_limits<double>::quiet_NaN();
            double v_r = std::numeric_limits<double>::quiet_NaN();
            if (id_pts.second.size() < 0) {
                continue;
            } else if (id_pts.second.size() == 1) {
            } else if (id_pts.second.size() == 2) {
                assert(id_pts.second[0].first == 0);
                assert(id_pts.second[1].first == 1);
                u_l = (id_pts.second[0].second)(3);
                v_l = (id_pts.second[0].second)(4);
                u_r = (id_pts.second[1].second)(3);
                v_r = (id_pts.second[1].second)(4);
                if(
                    std::isnan(u_l) || std::isnan(v_l) ||
                    std::isnan(u_r) || std::isnan(v_r) ||
                    std::isinf(u_l) || std::isinf(u_r) ||
                    std::isinf(v_l) || std::isinf(v_r)) {
                    continue;
                }
                cv::Point2f undistor_normalized_left_point =
                    cv::Point2f((id_pts.second[0].second)(0), (id_pts.second[0].second)(1));
                cv::Point2f undistor_normalized_right_point =
                    cv::Point2f((id_pts.second[1].second)(0), (id_pts.second[1].second)(1));
                cv::Point2f target_1, target_2;
                Eigen::Vector3d p0(undistor_normalized_left_point.x, undistor_normalized_left_point.y, 1);
                Eigen::Vector3d p1(undistor_normalized_right_point.x, undistor_normalized_right_point.y, 1);
                Eigen::Matrix3d Rc0c = stereo_camera_->camL_Pose_camLrect_.rotation().matrix(); // R1
                Eigen::Matrix3d Rc1c = stereo_camera_->camR_Pose_camRrect_.rotation().matrix(); // R2
                auto tmp0 = Rc0c.transpose() * p0;
                auto tmp1 = Rc1c.transpose() * p1;
                cv::Point2f p00(tmp0(0) / tmp0(2), tmp0(1) / tmp0(2));
                cv::Point2f p11(tmp1(0) / tmp1(2), tmp1(1) / tmp1(2));
                Eigen::Matrix3d K_ = stereo_calibration_->K().matrix();
                Utility::Normalized2Pixel(p00, &target_1, K_);
                Utility::Normalized2Pixel(p11, &target_2, K_);
                auto gap = abs(target_1.y - target_2.y);
                if(gap < 2) {
                    good++;
                    smart_stereo_measurements.push_back(
                        std::make_pair(landmark_id, gtsam::StereoPoint2(target_1.x, target_2.x, target_1.y)));
                } else if(gap < 10) {
                    average++;
                } else {
                    bad++;
                }
                total++;
                // cv::Point2f tmp_l, tmp_r;
                // Utility::Normalized2Pixel(undistor_normalized_left_point, &tmp_l, left_K_);
                // Utility::Normalized2Pixel(undistor_normalized_right_point, &tmp_r, right_K_);
                // cv::Point2f tmp_normal_l, tmp_normal_r;
                // Utility::Pixel2Normalized(tmp_l, &tmp_normal_l, K_);
                // Utility::Pixel2Normalized(tmp_r, &tmp_normal_r, K_);
                // cout<<"("<<u_l<<","<<v_l<<", "<<u_r<<","<<v_r<<") ";
                // cout<<"tmp_l("<<tmp_l.x<<","<<tmp_l.y<<", "<<tmp_r.x<<","<<tmp_r.y<<") ";
                // cout<<"target("<<target_1.x<<","<<target_1.y<<", "<<target_2.x<<","<<target_2.y<<") ";
                // cout<<"p0("<<p0(0)<<","<<p0(1)<<", "<<p1(0)<<","<<p1(1)<<") ";
                // cout<<"p00("<<p00.x<<","<<p00.y<<", "<<p11.x<<","<<p11.y<<") ";
                // cout<<endl;
            }
        }
        cout<<"(good:"<<good<<", average:"<<average<<", bad:"<<bad<<"; total:"<<total<<endl;
        Eigen::Matrix4d cur_T, prev_T, Tij;
        getPoseInWorldFrame(cur_T);
        getPoseInWorldFrame(frame_count - 1, prev_T);
        Tij = prev_T.inverse() * cur_T;
        boost::optional<gtsam::Pose3> pose_frome_last_frame(Tij);
        id_time_map_[curr_kf_id_] = curTime;
        time_id_map_[curTime] = curr_kf_id_;
        cout<<"addVisualInertial StateAndOptimize start:  "<<curr_kf_id_<<endl;
        TicToc t_gtsam_op;
        is_smoother_ok = addVisualInertialStateAndOptimize(
            int64_t(curTime), smart_stereo_measurements, pim, pose_frome_last_frame);
        if(is_smoother_ok) {
            updateStates(curr_kf_id_);
            printData();
            Eigen::Matrix4d new_T;
            getPoseInWorldFrame(new_T);
            if(curr_kf_id_ > 15) {
                motion_model_ = prev_T.inverse() * new_T;
            }
            if (1) {
                // Generate this map only if requested, since costly.
                // Also, if lmk type requested, fill lmk id to lmk type object.
                // WARNING this also cleans the lmks inside the old_smart_factors map!
                TicToc get_3dpoint;
                size_t kMinLmkObs = 2;
                lmk_ids_to_3d_points_in_time_horizon_ =
                    getMapLmkIdsTo3dPointsInTimeHorizon(
                        smoother_->getFactors(), kMinLmkObs);
                cout<<"get_3dpoint : "<<get_3dpoint.toc()<<";  point3.size()="
                    <<lmk_ids_to_3d_points_in_time_horizon_.size()<<endl;
            }
        } else {
            cout<<"faile to optimization, in id: "<<curr_kf_id_<<endl;
            assert(0);
        }
        static int gtsam_opt_index = 0;
        save_gtsam_op_times_ <<gtsam_opt_index<<","<< t_gtsam_op.toc()<<endl;
        cout << "gtsam_opt_index: " << gtsam_opt_index<<" [t_gtsam_op time: "<<t_gtsam_op.toc()<< ", t_solve time: "<<t_solve.toc()<<"]"<<std::endl;
        cout<<"addVisualInertial StateAndOptimize end:  "<<curr_kf_id_<<endl;
        gtsam_opt_index++;
        if (failureDetection()) {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }
        slideWindow();
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
        int64_t target_t = ceil(curTime) - backend_params_.horizon_;
        for(auto iter = time_id_map_.begin(), iter_next = time_id_map_.begin();
            iter != time_id_map_.end();
            iter = iter_next) {
            if(iter->first < target_t) {
                time_id_map_.erase(iter);
            }
            iter_next++;
        }
        auto iit = time_id_map_.begin();
        cout<<iit->first<<", "<<iit->second<<endl;
        cout<<endl;
    }
}

void Estimator::addInitialPriorFactors(const FrameId &frame_id) {
    // Set initial covariance for inertial factors
    // W_Pose_Blkf_ set by motion capture to start with
    gtsam::Matrix3 B_Rot_W = W_Pose_B_lkf_.rotation().matrix().transpose();
    // init params
    // int autoInitialize = 0;
    double initialPositionSigma = 0.00001;
    double initialRollPitchSigma = 10.0 / 180.0 * M_PI;
    double initialYawSigma = 0.1 / 180.0 * M_PI;
    double initialVelocitySigma = 1e-3;
    double initialAccBiasSigma = 0.1;
    double initialGyroBiasSigma = 0.01;

    // Set initial pose uncertainty: constrain mainly position and global yaw.
    // roll and pitch is observable, therefore low variance.
    gtsam::Matrix6 pose_prior_covariance = gtsam::Matrix6::Zero();
    pose_prior_covariance.diagonal()[0] =
        initialRollPitchSigma * initialRollPitchSigma;
    pose_prior_covariance.diagonal()[1] =
        initialRollPitchSigma * initialRollPitchSigma;
    pose_prior_covariance.diagonal()[2] = initialYawSigma * initialYawSigma;
    pose_prior_covariance.diagonal()[3] =
        initialPositionSigma * initialPositionSigma;
    pose_prior_covariance.diagonal()[4] =
        initialPositionSigma * initialPositionSigma;
    pose_prior_covariance.diagonal()[5] =
        initialPositionSigma * initialPositionSigma;
    // Rotate initial uncertainty into local frame, where the uncertainty is
    // specified.
    pose_prior_covariance.topLeftCorner(3, 3) =
        B_Rot_W * pose_prior_covariance.topLeftCorner(3, 3) *
        B_Rot_W.transpose();

    // step 1 : Add pose prior.
    // TODO(Toni): Make this noise model a member constant.
    gtsam::SharedNoiseModel noise_init_pose =
        gtsam::noiseModel::Gaussian::Covariance(pose_prior_covariance);
    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            gtsam::Symbol(kPoseSymbolChar, frame_id), W_Pose_B_lkf_,
            noise_init_pose));

    // step 2 : Add initial velocity priors.
    // TODO(Toni): Make this noise model a member constant.
    gtsam::SharedNoiseModel noise_init_vel_prior =
        gtsam::noiseModel::Isotropic::Sigma(3, initialVelocitySigma);
    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            gtsam::Symbol(kVelocitySymbolChar, frame_id), W_Vel_B_lkf_,
            noise_init_vel_prior));

    // step 3 : Add initial bias priors:
    gtsam::Vector6 prior_biasSigmas;
    prior_biasSigmas.head<3>().setConstant(initialAccBiasSigma);
    prior_biasSigmas.tail<3>().setConstant(initialGyroBiasSigma);
    // TODO(Toni): Make this noise model a member constant.
    gtsam::SharedNoiseModel imu_bias_prior_noise =
        gtsam::noiseModel::Diagonal::Sigmas(prior_biasSigmas);
    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
            gtsam::Symbol(kImuBiasSymbolChar, frame_id), imu_bias_lkf_,
            imu_bias_prior_noise));
}

bool Estimator::initStateAndSetPriors(
    const VioNavStateTimestamped &vio_nav_state_initial_seed) {
    // Clean state
    new_values_.clear();
    // Update member variables.
    timestamp_lkf_ = vio_nav_state_initial_seed.timestamp_;
    W_Pose_B_lkf_ = vio_nav_state_initial_seed.pose_;
    W_Vel_B_lkf_ = vio_nav_state_initial_seed.velocity_;
    imu_bias_lkf_ = vio_nav_state_initial_seed.imu_bias_;
    std::cout << "Initial state seed: \n"
              << " - Initial timestamp: " << timestamp_lkf_ << '\n'
              << " - Initial pose: " << W_Pose_B_lkf_ << '\n'
              << " - Initial vel: " << W_Vel_B_lkf_.transpose() << '\n'
              << " - Initial IMU bias: " << imu_bias_lkf_
              << std::endl;
    // Can't add inertial prior factor until we have a state measurement.
    addInitialPriorFactors(curr_kf_id_);
    // Add initial state seed
    new_values_.insert(gtsam::Symbol(kPoseSymbolChar, curr_kf_id_),
                       W_Pose_B_lkf_);
    new_values_.insert(gtsam::Symbol(kVelocitySymbolChar, curr_kf_id_),
                       W_Vel_B_lkf_);
    new_values_.insert(gtsam::Symbol(kImuBiasSymbolChar, curr_kf_id_),
                       imu_bias_lkf_);
    return optimize(vio_nav_state_initial_seed.timestamp_, curr_kf_id_,
                    backend_params_.numOptimize_);
}

bool Estimator::addVisualInertialStateAndOptimize(
    const Timestamp &timestamp_kf_sec,
    const StereoMeasurements &status_smart_stereo_measurements_kf,
    const std::shared_ptr<gtsam::PreintegratedImuMeasurements> &pim,
    boost::optional<gtsam::Pose3> pose_frome_last_frame) {
    TicToc procee_factors_T;
    addImuValues(curr_kf_id_, pim);
    addImuFactor(last_kf_id_, curr_kf_id_, pim);
    if (pose_frome_last_frame) {
        addBetweenFactor(last_kf_id_, curr_kf_id_, *pose_frome_last_frame);
    }
    LandmarkIds landmarks_kf;
    addStereoMeasurementsToFeatureTracks(
        curr_kf_id_, status_smart_stereo_measurements_kf, &landmarks_kf);
    if (!keyframe_) {
        addZeroVelocityPrior(curr_kf_id_);
        addNoMotionFactor(last_kf_id_, curr_kf_id_);
    } else {
        addLandmarksToGraph(landmarks_kf);
    }
    return optimize(timestamp_kf_sec, curr_kf_id_,
                    backend_params_.numOptimize_);
}

void Estimator::addImuValues(
    const FrameId &cur_id,
    const std::shared_ptr<gtsam::PreintegratedImuMeasurements> &pim) {
    gtsam::NavState navstate_lkf(W_Pose_B_lkf_, W_Vel_B_lkf_);
    gtsam::NavState navstate_k = pim->predict(navstate_lkf, imu_bias_lkf_);
    new_values_.insert(gtsam::Symbol(kPoseSymbolChar, cur_id),
                       navstate_k.pose());
    new_values_.insert(gtsam::Symbol(kVelocitySymbolChar, cur_id),
                       navstate_k.velocity());
    new_values_.insert(gtsam::Symbol(kImuBiasSymbolChar, cur_id),
                       imu_bias_lkf_);
}

void Estimator::addImuFactor(
    const FrameId &from_id,
    const FrameId &to_id,
    const std::shared_ptr<gtsam::PreintegratedImuMeasurements> &pim) {
    // new_imu_prior_and_other_factors_.push_back(
    //     boost::make_shared<gtsam::CombinedImuFactor>(
    //         gtsam::Symbol(kPoseSymbolChar, from_id),
    //         gtsam::Symbol(kVelocitySymbolChar, from_id),
    //         gtsam::Symbol(kPoseSymbolChar, to_id),
    //         gtsam::Symbol(kVelocitySymbolChar, to_id),
    //         gtsam::Symbol(kImuBiasSymbolChar, from_id),
    //         gtsam::Symbol(kImuBiasSymbolChar, to_id), *pim));

    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::ImuFactor>(
            gtsam::Symbol(kPoseSymbolChar, from_id),
            gtsam::Symbol(kVelocitySymbolChar, from_id),
            gtsam::Symbol(kPoseSymbolChar, to_id),
            gtsam::Symbol(kVelocitySymbolChar, to_id),
            gtsam::Symbol(kImuBiasSymbolChar, from_id),
            *pim));
    static const gtsam::imuBias::ConstantBias zero_bias(
            gtsam::Vector3(0.0, 0.0, 0.0), gtsam::Vector3(0.0, 0.0, 0.0));
    // Factor to discretize and move normalize by the interval between measurements:
    CHECK_NE(imu_params_.nominal_sampling_time_s_, 0.0) << "Nominal IMU sampling time cannot be 0 s.";
    // See Trawny05 http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf   Eq. 130
    const double& sqrt_delta_t_ij = std::sqrt(pim->deltaTij());
    gtsam::Vector6 bias_sigmas;
    bias_sigmas.head<3>().setConstant(sqrt_delta_t_ij * imu_params_.acc_random_walk_);
    bias_sigmas.tail<3>().setConstant(sqrt_delta_t_ij * imu_params_.gyro_random_walk_);
    const gtsam::SharedNoiseModel& bias_noise_model = gtsam::noiseModel::Diagonal::Sigmas(bias_sigmas);
    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
            gtsam::Symbol(kImuBiasSymbolChar, from_id),
            gtsam::Symbol(kImuBiasSymbolChar, to_id),
            zero_bias,
            bias_noise_model));
}

void Estimator::addBetweenFactor(const FrameId &from_id,
                                 const FrameId &to_id,
                                 const gtsam::Pose3 &from_id_POSE_to_id) {
    gtsam::Vector6 precisions;
    precisions.head<3>().setConstant(backend_params_.betweenRotationPrecision_);
    precisions.tail<3>().setConstant(
        backend_params_.betweenTranslationPrecision_);
    const gtsam::SharedNoiseModel &betweenNoise_ =
        gtsam::noiseModel::Diagonal::Precisions(precisions);
    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            gtsam::Symbol(kPoseSymbolChar, from_id),
            gtsam::Symbol(kPoseSymbolChar, to_id), from_id_POSE_to_id,
            betweenNoise_));
}

void Estimator::addStereoMeasurementsToFeatureTracks(
    const int &curr_kf_id,
    const StereoMeasurements &stereo_meas_kf,
    LandmarkIds *landmarks_kf) {
    CHECK_NOTNULL(landmarks_kf);
    const size_t &n_stereo_measurements = stereo_meas_kf.size();
    landmarks_kf->resize(n_stereo_measurements);
    for (size_t i = 0u; i < n_stereo_measurements; ++i) {
        const LandmarkId &lmk_id_in_kf_i = stereo_meas_kf[i].first;
        const StereoPoint2 &stereo_px_i = stereo_meas_kf[i].second;
        CHECK_NE(lmk_id_in_kf_i, -1) << "landmarkId_kf_i == -1?";
        if(lmk_id_in_kf_i != 0) {
            DCHECK(std::find(landmarks_kf->begin(),
                        landmarks_kf->end(),
                        lmk_id_in_kf_i) == landmarks_kf->end());
        }
        (*landmarks_kf)[i] = lmk_id_in_kf_i;
        const FeatureTracks::iterator &feature_track_it =
            feature_tracks_.find(lmk_id_in_kf_i);
        if (feature_track_it == feature_tracks_.end()) {
            feature_tracks_.insert(std::make_pair(
                lmk_id_in_kf_i, FeatureTrack(curr_kf_id, stereo_px_i)));
            ++landmark_count_;
        } else {
            feature_track_it->second.obs_.push_back(
                std::make_pair(curr_kf_id, stereo_px_i));
        }
    }
}

void Estimator::addLandmarksToGraph(const LandmarkIds &landmarks_kf) {
    int n_new_landmarks = 0;
    int n_updated_landmarks = 0;
    for (const LandmarkId &lmk_id : landmarks_kf) {
        FeatureTrack &ft = feature_tracks_.at(lmk_id);
        if (ft.obs_.size() < 2) {
            continue;
        }
        if (!ft.in_ba_graph_) {
            ft.in_ba_graph_ = true;
            addLandmarkToGraph(lmk_id, ft);
            ++n_new_landmarks;
        } else {
            const std::pair<FrameId, StereoPoint2> obs_kf = ft.obs_.back();
            updateLandmarkInGraph(lmk_id, obs_kf);
            ++n_updated_landmarks;
        }
    }
}

void Estimator::addLandmarkToGraph(const LandmarkId &lmk_id,
                                   const FeatureTrack &ft) {
    SmartStereoFactor::shared_ptr new_factor =
        boost::make_shared<SmartStereoFactor>(
            smart_noise_, smart_factors_params_, B_Pose_leftCam_);
    for (const std::pair<FrameId, StereoPoint2> &obs : ft.obs_) {
        const FrameId &frame_id = obs.first;
        if(ceil(curTime) - floor(id_time_map_[frame_id]) < backend_params_.horizon_) {
            const gtsam::Symbol &pose_symbol = gtsam::Symbol(kPoseSymbolChar, frame_id);
            const StereoPoint2 &measurement = obs.second;
            new_factor->add(measurement, pose_symbol, stereo_calibration_);
        }
    }
    new_smart_factors_.insert(std::make_pair(lmk_id, new_factor));
    old_smart_factors_.insert(
        std::make_pair(lmk_id, std::make_pair(new_factor, -1)));
}

void Estimator::updateLandmarkInGraph(
    const LandmarkId &lmk_id,
    const std::pair<FrameId, StereoPoint2> &new_measurement) {
    auto old_smart_factors_it = old_smart_factors_.find(lmk_id);
    CHECK(old_smart_factors_it != old_smart_factors_.end())
        << "Landmark not found in old_smart_factors_ with id: " << lmk_id;
    SmartStereoFactor::shared_ptr new_factor =
        boost::make_shared<SmartStereoFactor>(smart_noise_, smart_factors_params_, B_Pose_leftCam_);
    FeatureTrack ft = feature_tracks_.at(lmk_id);
    for (const std::pair<FrameId, StereoPoint2> &obs : ft.obs_) {
        const FrameId &frame_id = obs.first;
        // ceil 像上取整数
        if(ceil(curTime) - floor(id_time_map_[frame_id]) < backend_params_.horizon_) {
            // kPoseSymbolChar : T_w_imu
            const gtsam::Symbol &pose_symbol = gtsam::Symbol(kPoseSymbolChar, frame_id);
            const StereoPoint2 &measurement = obs.second;
            new_factor->add(measurement, pose_symbol, stereo_calibration_);
        }
    }
    Slot slot = old_smart_factors_it->second.second;
    if (slot != -1) {
        new_smart_factors_.insert(std::make_pair(lmk_id, new_factor));
    } else {
        FeatureTrack ft = feature_tracks_.at(lmk_id);
        cout<< "ft.obs_.size(): "<<ft.obs_.size()<<endl;
        cout<< "When updating the smart factor, its slot should not be -1!"
               " Offensive lmk_id: "
            << lmk_id;
    }
    old_smart_factors_it->second.first = new_factor;
}

void Estimator::addZeroVelocityPrior(const FrameId &frame_id) {
    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            gtsam::Symbol(kVelocitySymbolChar, frame_id),
            gtsam::Vector3::Zero(), zero_velocity_prior_noise_));
}

void Estimator::addNoMotionFactor(const FrameId &from_id,
                                  const FrameId &to_id) {
    new_imu_prior_and_other_factors_.push_back(
        boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            gtsam::Symbol(kPoseSymbolChar, from_id),
            gtsam::Symbol(kPoseSymbolChar, to_id), gtsam::Pose3::identity(),
            no_motion_prior_noise_));
}

bool Estimator::optimize(
    const Timestamp &timestamp_kf_sec,
    const FrameId &cur_id,
    const size_t &max_extra_iterations,
    const gtsam::FactorIndices &extra_factor_slots_to_delete) {
    TicToc optimize_T;
    size_t new_smart_factors_size = new_smart_factors_.size();
    gtsam::FactorIndices delete_slots = extra_factor_slots_to_delete;
    gtsam::NonlinearFactorGraph new_factors_tmp;
    new_factors_tmp.reserve(new_smart_factors_size +
                            new_imu_prior_and_other_factors_.size());
    std::map<gtsam::FactorIndex, LandmarkId> newFactor2lm;
    int index = 0;
    for (const auto &new_smart_factor : new_smart_factors_) {
        LandmarkId lmk_id = new_smart_factor.first;  // don't use &
        const auto &old_smart_factor_it = old_smart_factors_.find(lmk_id);
        Slot slot = old_smart_factor_it->second.second;
        if (slot != -1) {
            if (!smoother_->getFactors().exists(slot)) {
                old_smart_factors_.erase(old_smart_factor_it);
                CHECK(deleteLmkFromFeatureTracks(lmk_id));
            } else {
                delete_slots.push_back(slot);
                new_factors_tmp.push_back(new_smart_factor.second);
                newFactor2lm[index++] = lmk_id;
                DCHECK(boost::dynamic_pointer_cast<SmartStereoFactor>(
                    new_smart_factor.second));
            }
        } else {
            newFactor2lm[index++] = lmk_id;
            new_factors_tmp.push_back(new_smart_factor.second);
            DCHECK(boost::dynamic_pointer_cast<SmartStereoFactor>(
                    new_smart_factor.second));
        }
    }
    new_factors_tmp.push_back(new_imu_prior_and_other_factors_.begin(),
                              new_imu_prior_and_other_factors_.end());
    std::map<Key, double> timestamps;
    BOOST_FOREACH (const gtsam::Values::ConstKeyValuePair &key_value,
                   new_values_) {
        timestamps[key_value.key] =
            timestamp_kf_sec;
    }
    DCHECK_EQ(timestamps.size(), new_values_.size());
    Smoother::Result result;
    TicToc updateSmoother_1;
    bool is_smoother_ok = updateSmoother(&result, new_factors_tmp, new_values_,
                                         timestamps, delete_slots);
    printf("update Smoother time: %f\n", updateSmoother_1.toc());
    new_smart_factors_.clear();
    new_imu_prior_and_other_factors_.resize(0);
    new_values_.clear();
    // updateNewSmartFactorsSlots()
    cout<<"smoother_->getISAM2Result().newFactorsIndices.size()="<<smoother_->getISAM2Result().newFactorsIndices.size()
    <<", newFactor2lm.size()="<<newFactor2lm.size()<<endl;
    // cout<<"(";
    for (const auto &f2l : newFactor2lm) {
        const auto &it = old_smart_factors_.find(f2l.second);
        if(it == old_smart_factors_.end()) {
            continue;
        }
        // cout<<"("<<f2l.first;
        auto slot = smoother_->getISAM2Result().newFactorsIndices.at(f2l.first);
        it->second.second = slot;
        // cout<<", "<<(uint64_t)slot<<") ";
        it->second.first = boost::dynamic_pointer_cast<SmartStereoFactor>(
                    smoother_->getFactors().at(slot));
        boost::shared_ptr<SmartStereoFactor> gsf =
            boost::dynamic_pointer_cast<SmartStereoFactor>(smoother_->getFactors().at(slot));

    }
    // cout<<")"<<endl;
    cout<<"optimize_T: "<<optimize_T.toc()<<endl;
    return is_smoother_ok;
}

bool Estimator::updateSmoother(Smoother::Result *result,
                               const gtsam::NonlinearFactorGraph &new_factors,
                               const gtsam::Values &new_values,
                               const std::map<Key, double> &timestamps,
                               const gtsam::FactorIndices &delete_slots) {
    Smoother smoother_backup(*smoother_);
    bool got_cheirality_exception = false;
    gtsam::Symbol lmk_symbol_cheirality;
    try {
        *result = smoother_->update(new_factors, new_values, timestamps,
                                    delete_slots);
    }
    catch (const gtsam::IndeterminantLinearSystemException& e) {
        printf(" IndeterminantLinearSystemException \n");
        std::cout << e.what();
        const gtsam::Key& var = e.nearbyVariable();
        gtsam::Symbol symb(var);
        std::cout << "ERROR: Variable has type '" << symb.chr() << "' "
                << "and index " << symb.index() << std::endl;
        // state_.print();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::InvalidNoiseModel& e) {
        printf(" InvalidNoiseModel \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::InvalidMatrixBlock& e) {
        printf(" InvalidMatrixBlock \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::InvalidDenseElimination& e) {
        printf(" InvalidDenseElimination \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::InvalidArgumentThreadsafe& e) {
        printf(" InvalidArgumentThreadsafe \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::ValuesKeyDoesNotExist& e) {
        printf(" ValuesKeyDoesNotExist \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::CholeskyFailed& e) {
        printf(" CholeskyFailed \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::CheiralityException& e) {
        printf(" CheiralityException \n");
        std::cout << e.what();
        const gtsam::Key& lmk_key = e.nearbyVariable();
        lmk_symbol_cheirality = gtsam::Symbol(lmk_key);
        std::cout << "ERROR: Variable has type '" << lmk_symbol_cheirality.chr()
                << "' "
                << "and index " << lmk_symbol_cheirality.index();
        printSmootherInfo(new_factors, delete_slots);
        got_cheirality_exception = true;
    } catch (const gtsam::StereoCheiralityException& e) {
        printf(" StereoCheiralityException \n");
        std::cout << e.what();
        const gtsam::Key& lmk_key = e.nearbyVariable();
        lmk_symbol_cheirality = gtsam::Symbol(lmk_key);
        std::cout << "ERROR: Variable has type '" << lmk_symbol_cheirality.chr()
                << "' "
                << "and index " << lmk_symbol_cheirality.index();
        printSmootherInfo(new_factors, delete_slots);
        got_cheirality_exception = true;
    } catch (const gtsam::RuntimeErrorThreadsafe& e) {
        printf(" RuntimeErrorThreadsafe \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const gtsam::OutOfRangeThreadsafe& e) {
        printf(" OutOfRangeThreadsafe \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const std::out_of_range& e) {
        printf(" out_of_range \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (const std::exception& e) {
        // Catch anything thrown within try block that derives from
        // std::exception.
        printf(" exception \n");
        std::cout << e.what();
        printSmootherInfo(new_factors, delete_slots);
        return false;
    } catch (...) {
        // Catch the rest of exceptions.
        std::cout << "Unrecognized exception.";
        printSmootherInfo(new_factors, delete_slots);
        return false;
    }
    // bool FLAGS_process_cheirality = true;
    bool FLAGS_process_cheirality = false;
    int FLAGS_max_number_of_cheirality_exceptions = 5;
    // cheirality constraint, 判断特征点是否在相机后面 一般可以用来判断匹配点对是否正确，
    // 并经常用于从本质矩阵 E E E的四组分解中筛选出正确的一组 R , t R,t R,t
    // /** return the cheirality status flag */
    // bool isPointBehindCamera() const { return result_.behindCamera(); }
    if (FLAGS_process_cheirality) {
        if (got_cheirality_exception) {
            std::cout << "Starting processing cheirality exception: " << counter_of_exceptions_ << std::endl;
            counter_of_exceptions_++;
            // Restore smoother as it was before failure.
            *smoother_ = smoother_backup;
            // Limit the number of cheirality exceptions per run.
            CHECK_LE(counter_of_exceptions_,
                       FLAGS_max_number_of_cheirality_exceptions);
            // Check that we have a landmark.
            CHECK_EQ(lmk_symbol_cheirality.chr(), 'l');
            // Now that we know the lmk id, delete all factors attached to it!
            gtsam::NonlinearFactorGraph new_factors_tmp_cheirality;
            gtsam::Values new_values_cheirality;
            std::map<Key, double> timestamps_cheirality;
            gtsam::FactorIndices delete_slots_cheirality;
            const gtsam::NonlinearFactorGraph graph = smoother_->getFactors();
            std::cout << "Starting cleanCheiralityLmk..." << std::endl;
            cleanCheiralityLmk(lmk_symbol_cheirality,
                                &new_factors_tmp_cheirality,
                                &new_values_cheirality,
                                &timestamps_cheirality,
                                &delete_slots_cheirality,
                                graph,
                                new_factors,
                                new_values,
                                timestamps,
                                delete_slots);
            // Try again to Optimize. This is a recursive call.
            cout << "Starting update Smoother after handling "
                            "cheirality exception.";
            bool status = updateSmoother(
                result, new_factors_tmp_cheirality, new_values_cheirality,
                timestamps_cheirality, delete_slots_cheirality);
            cout << "Finished update Smoother after handling "
                            "cheirality exception";
            return status;
        } else {
            counter_of_exceptions_ = 0;
        }
    }
    return true;
}

void Estimator::printSmootherInfo(
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::FactorIndices& delete_slots,
    const std::string& message,
    const bool& showDetails) const {
    printf("!!!!!!!!!!iSAM2 optimazation Error!!  \n");
    assert(0);
}

void Estimator::updateStates(const FrameId &cur_id) {
    auto values = smoother_->getLinearizationPoint();
    cout<<"values.size() = "<<values.size();
    // get factorGraph
    auto factors = smoother_->getFactors();
    int num_ = 0;
    for(auto factor : factors) {
        num_++;
    }
    static int factor_index = 0;
    save_total_factors_ <<factor_index<<","<< factors.size()<<endl;
    save_total_nonnull_factors_ <<factor_index<<","<< factors.nrFactors()<<endl;
    factor_index++;
    cout << "Iterator factors: " <<  num_ << endl;
    cout << "(Total factors: " <<  factors.size();
    cout << ", Non-null factors: " <<  factors.nrFactors() <<") "<< endl;
    // iSAM2 Smoother Keys size
    int size = 0;
    for(const gtsam::FixedLagSmoother::KeyTimestampMap::value_type& key_timestamp:  smoother_->timestamps()) {
        size++;
    }
    cout << " iSAM2 Smoother Keys size =  "<<size << endl;
    // get value after update
    state_ = smoother_->calculateEstimate();
    DCHECK(state_.find(gtsam::Symbol(kPoseSymbolChar, cur_id)) != state_.end());
    DCHECK(state_.find(gtsam::Symbol(kVelocitySymbolChar, cur_id)) !=
           state_.end());
    DCHECK(state_.find(gtsam::Symbol(kImuBiasSymbolChar, cur_id)) !=
           state_.end());
    W_Pose_B_lkf_ =
        state_.at<gtsam::Pose3>(gtsam::Symbol(kPoseSymbolChar, cur_id));
    W_Vel_B_lkf_ =
        state_.at<gtsam::Vector3>(gtsam::Symbol(kVelocitySymbolChar, cur_id));
    imu_bias_lkf_ = state_.at<gtsam::imuBias::ConstantBias>(
        gtsam::Symbol(kImuBiasSymbolChar, cur_id));
    // update imu frontend
    keyframe_imu_->resetIntegrationWithCachedBias(imu_bias_lkf_);
    {
        Rs[WINDOW_SIZE] = W_Pose_B_lkf_.rotation().matrix();
        Ps[WINDOW_SIZE] = W_Pose_B_lkf_.translation();
        if (USE_IMU) {
            Vs[WINDOW_SIZE] = Eigen::Vector3d(W_Vel_B_lkf_);
            Bas[WINDOW_SIZE] = imu_bias_lkf_.accelerometer();
            Bgs[WINDOW_SIZE] = imu_bias_lkf_.gyroscope();
        }
    }
}

bool Estimator::deleteLmkFromFeatureTracks(const LandmarkId &lmk_id) {
    if (feature_tracks_.find(lmk_id) != feature_tracks_.end()) {
        feature_tracks_.erase(lmk_id);
        return true;
    }
    return false;
}


void Estimator::updateNewSmartFactorsSlots(
    const std::vector<LandmarkId> &lmk_ids_of_new_smart_factors,
    SmartFactorMap *old_smart_factors) {
    CHECK_NOTNULL(old_smart_factors);
    if(lmk_ids_of_new_smart_factors.size() <= 0) {
        return;
    }
            int num = 0;
            cout<<"lmk_ids_of_new_smart_factors:  ";
            for(auto l : lmk_ids_of_new_smart_factors) {
                num++;
                cout<<l<<" ";
            }
            cout<<" (size: "<<num<<endl;
            const gtsam::ISAM2Result &result = smoother_->getISAM2Result();
            num = 0;
            cout<<"newFactorsIndices:  ";
            for(auto l : result.newFactorsIndices) {
                cout<<l<<" ";
                num++;
            }
            cout<<" (size: "<<num<<endl;
    for (size_t i = 0u; i < lmk_ids_of_new_smart_factors.size(); ++i) {
        // if smart_factors_params->setEnableEPI(false); here will throw exception
        DCHECK(i < result.newFactorsIndices.size())
            << "There are more new smart factors than new factors added to the "
               "graph."<<"  i: "<<i<<" result.newFactorsIndices.size(): "<<result.newFactorsIndices.size() << 
               "  lmk_ids_of_new_smart_factors.size(): "<<lmk_ids_of_new_smart_factors.size();
        // Get new slot in the graph for the newly added smart factor.
        const size_t &slot = result.newFactorsIndices.at(i);
        cout<<" ("<<i<<", "<<slot<<")";

        // TODO this will not work if there are non-smart factors!!!
        // Update slot using isam2 indices.
        // ORDER of inclusion of factors in the ISAM2::update() function
        // matters, as these indices have a 1-to-1 correspondence with the
        // factors.

        // BOOKKEEPING, for next iteration to know which slots have to be
        // deleted before adding the new smart factors. Find the entry in
        // old_smart_factors_.
        const auto &it =
            old_smart_factors->find(lmk_ids_of_new_smart_factors.at(i));
        DCHECK(it != old_smart_factors->end())
            << "Trying to access unavailable factor.";
        // CHECK that the factor in the graph at slot position is a smart
        // factor.
        if(!smoother_->getFactors().exists(slot)) {
            cout << "!!!!!!!!!!!!!!!!!!";
            assert(0);
        }
        if(smoother_->getFactors().at(slot)) {
            // cout<<"(" << typeid(smoother_->getFactors().at(slot)).name()<<"), ";
        } else {
            cout<<"null factors!!!: "<<"(" << typeid(smoother_->getFactors().at(slot)).name()<<"), ";
            return;
        }

        DCHECK(boost::dynamic_pointer_cast<SmartStereoFactor>(
            smoother_->getFactors().at(slot)));
        // yhh
        // CHECK that shared ptrs point to the same smart factor.
        // make sure no one is cloning SmartSteroFactors.
        DCHECK_EQ(it->second.first,
                  boost::dynamic_pointer_cast<SmartStereoFactor>(
                      smoother_->getFactors().at(slot)))
            << "Non-matching addresses for same factors for lmk with id: "
            << lmk_ids_of_new_smart_factors.at(i) << " in old_smart_factors_ "
            << "VS factor in graph at slot: " << slot
            << ". Slot previous to update was: " << it->second.second<<endl
            <<"  typeid:  " << typeid(smoother_->getFactors().at(slot)).name()
            <<"  typeid:  " << typeid(it->second.first).name();
        it->second.second = slot;
    }
}

void Estimator::cleanCheiralityLmk(
    const gtsam::Symbol& lmk_symbol,
    gtsam::NonlinearFactorGraph* new_factors_tmp_cheirality,
    gtsam::Values* new_values_cheirality,
    std::map<Key, double>* timestamps_cheirality,
    gtsam::FactorIndices* delete_slots_cheirality,
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::Values& new_values,
    const std::map<Key, double>& timestamps,
    const gtsam::FactorIndices& delete_slots) {
  CHECK_NOTNULL(new_factors_tmp_cheirality);
  CHECK_NOTNULL(new_values_cheirality);
  CHECK_NOTNULL(timestamps_cheirality);
  CHECK_NOTNULL(delete_slots_cheirality);
  const gtsam::Key& lmk_key = lmk_symbol.key();
  // Delete from new factors.
  VLOG(10) << "Starting delete from new factors...";
  deleteAllFactorsWithKeyFromFactorGraph(
      lmk_key, new_factors_tmp, new_factors_tmp_cheirality);
  VLOG(10) << "Finished delete from new factors.";

  // Delete from new values.
  VLOG(10) << "Starting delete from new values...";
  bool is_deleted_from_values =
      deleteKeyFromValues(lmk_key, new_values, new_values_cheirality);
  VLOG(10) << "Finished delete from timestamps.";

  // Delete from new values.
  VLOG(10) << "Starting delete from timestamps...";
  bool is_deleted_from_timestamps =
      deleteKeyFromTimestamps(lmk_key, timestamps, timestamps_cheirality);
  VLOG(10) << "Finished delete from timestamps.";

  // Check that if we deleted from values, we should have deleted as well
  // from timestamps.
  CHECK_EQ(is_deleted_from_values, is_deleted_from_timestamps);

  // Delete slots in current graph.
  VLOG(10) << "Starting delete from current graph...";
  *delete_slots_cheirality = delete_slots;
  std::vector<size_t> slots_of_extra_factors_to_delete;
  // Achtung: This has the chance to make the plane underconstrained, if
  // we delete too many point_plane factors.
  findSlotsOfFactorsWithKey(lmk_key, graph, &slots_of_extra_factors_to_delete);
  delete_slots_cheirality->insert(delete_slots_cheirality->end(),
                                  slots_of_extra_factors_to_delete.begin(),
                                  slots_of_extra_factors_to_delete.end());
  VLOG(10) << "Finished delete from current graph.";

  //////////////////////////// BOOKKEEPING
  ////////////////////////////////////////
  const LandmarkId& lmk_id = lmk_symbol.index();

  // Delete from feature tracks.
  VLOG(10) << "Starting delete from feature tracks...";
  CHECK(deleteLmkFromFeatureTracks(lmk_id));
  VLOG(10) << "Finished delete from feature tracks.";

  // Delete from extra structures (for derived classes).
  VLOG(10) << "Starting delete from extra structures...";
  std::cout<< "There is nothing to delete for lmk with id: " << lmk_id;
  VLOG(10) << "Finished delete from extra structures.";
}

void Estimator::deleteAllFactorsWithKeyFromFactorGraph(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& factor_graph,
    gtsam::NonlinearFactorGraph* factor_graph_output) {
  CHECK_NOTNULL(factor_graph_output);
  size_t new_factors_slot = 0;
  *factor_graph_output = factor_graph;
  for (auto it = factor_graph_output->begin();
       it != factor_graph_output->end();) {
    if (*it) {
      if ((*it)->find(key) != (*it)->end()) {
        // We found our lmk in the list of keys of the factor.
        // Sanity check, this lmk has no priors right?
        CHECK(!boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Point3>>(
            *it));
        // We are not deleting a smart factor right?
        // Otherwise we need to update structure:
        // lmk_ids_of_new_smart_factors...
        CHECK(!boost::dynamic_pointer_cast<SmartStereoFactor>(*it));
        // Whatever factor this is, it has our lmk...
        // Delete it.
        // std::cout << "Delete factor in new_factors at slot # "
        //              << new_factors_slot << " of new_factors graph.";
        it = factor_graph_output->erase(it);
      } else {
        it++;
      }
    } else {
      std::cout << "*it, which is itself a pointer, is null." << std::endl;
      it++;
    }
    new_factors_slot++;
  }
}

// Returns if the key in timestamps could be removed or not.
bool Estimator::deleteKeyFromValues(const gtsam::Key& key,
                                     const gtsam::Values& values,
                                     gtsam::Values* values_output) {
  CHECK_NOTNULL(values_output);
  *values_output = values;
  if (values.find(key) != values.end()) {
    // We found the lmk in new values, delete it.
    LOG(WARNING) << "Delete value in new_values for key "
                 << gtsam::DefaultKeyFormatter(key);
    CHECK(values_output->find(key) != values_output->end());
    try {
      values_output->erase(key);
    } catch (const gtsam::ValuesKeyDoesNotExist& e) {
      LOG(FATAL) << e.what();
    } catch (...) {
      LOG(FATAL) << "Unhandled exception when erasing key"
                    " in new_values_cheirality";
    }
    return true;
  }
  return false;
}

// Returns if the key in timestamps could be removed or not.
bool Estimator::deleteKeyFromTimestamps(
    const gtsam::Key& key,
    const std::map<Key, double>& timestamps,
    std::map<Key, double>* timestamps_output) {
  CHECK_NOTNULL(timestamps_output);
  *timestamps_output = timestamps;
  if (timestamps_output->find(key) != timestamps_output->end()) {
    timestamps_output->erase(key);
    return true;
  }
  return false;
}

// Returns if the key in timestamps could be removed or not.
void Estimator::findSlotsOfFactorsWithKey(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& graph,
    std::vector<size_t>* slots_of_factors_with_key) {
  CHECK_NOTNULL(slots_of_factors_with_key);
  slots_of_factors_with_key->resize(0);
  size_t slot = 0;
  for (const boost::shared_ptr<gtsam::NonlinearFactor>& g : graph) {
    if (g) {
      // Found a valid factor.
      if (g->find(key) != g->end()) {
        // Whatever factor this is, it has our lmk...
        // Sanity check, this lmk has no priors right?
        CHECK(!boost::dynamic_pointer_cast<gtsam::LinearContainerFactor>(g));
        CHECK(
            !boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Point3>>(g));
        // Sanity check that we are not deleting a smart factor.
        CHECK(!boost::dynamic_pointer_cast<SmartStereoFactor>(g));
        // Delete it.
        LOG(WARNING) << "Delete factor in graph at slot # " << slot
                     << " corresponding to lmk with id: "
                     << gtsam::Symbol(key).index();
        CHECK(graph.exists(slot));
        slots_of_factors_with_key->push_back(slot);
      }
    }
    slot++;
  }
}

// Get valid 3D points and corresponding lmk id.
// Warning! it modifies old_smart_factors_!!
std::unordered_map<LandmarkId, gtsam::Point3> Estimator::getMapLmkIdsTo3dPointsInTimeHorizon(
    const gtsam::NonlinearFactorGraph& graph,
    const size_t& min_age) {
    std::unordered_map<LandmarkId, gtsam::Point3> points_with_id;
    // Step 1: Add landmarks encoded in the smart factors.
    // old_smart_factors_ has all smart factors included so far.
    // Retrieve lmk ids from smart factors in state.
    size_t nr_valid_smart_lmks = 0, nr_smart_lmks = 0;
    for (SmartFactorMap::iterator old_smart_factor_it = old_smart_factors_.begin();
        old_smart_factor_it != old_smart_factors_.end(); old_smart_factor_it++) {
        //!< landmarkId -> {SmartFactorPtr, SlotIndex}
        // Store number of smart lmks (one smart factor per landmark).
        nr_smart_lmks++;
        // Retrieve lmk_id of the smart factor.
        const LandmarkId& lmk_id = old_smart_factor_it->first;
        // Retrieve smart factor.
        const SmartStereoFactor::shared_ptr& smart_factor_ptr = old_smart_factor_it->second.first;
        // Check that pointer is well definied.
        CHECK(smart_factor_ptr) << "Smart factor is not well defined.";
        // Retrieve smart factor slot in the graph.
        const Slot& slot_id = old_smart_factor_it->second.second;
        // Check that slot is admissible.
        DCHECK(slot_id >= 0) << "Slot of smart factor is not admissible.";
        // Ensure the graph size is small enough to cast to int.
        DCHECK_LT(graph.size(), std::numeric_limits<Slot>::max()) << "Invalid cast, that would cause an overflow!";
        // Slot should be inferior to the size of the graph.
        DCHECK_LT(slot_id, static_cast<Slot>(graph.size()));
        // Check that this slot_id exists in the graph, aka check that it is
        // in bounds and that the pointer is live (aka at(slot_id) works).
        if (!graph.exists(slot_id)) {
            // cout << "The slot: " << slot_id << " does not exist in the graph. lmk id: " << lmk_id <<endl;
            old_smart_factor_it = old_smart_factors_.erase(old_smart_factor_it);
            // Update as well the feature track....
            // TODO(TONI): please remove this and centralize how feature tracks
            // and new/old_smart_factors are added and removed!
            CHECK(deleteLmkFromFeatureTracks(lmk_id));
            continue;
        } else {
            // cout << "Slot id: " << slot_id << " for smart factor of lmk id: " << lmk_id<<endl;
        }
        // Check that the pointer smart_factor_ptr points to the right element in the graph.
        if (smart_factor_ptr != graph.at(slot_id)) {
            // Pointer in the graph does not match
            // the one we stored in old_smart_factors_
            // ERROR: if the pointers don't match, then the code that follows does
            // not make any sense, since we are using lmk_id which comes from
            // smart_factor and result which comes from graph[slot_id], we should
            // use smart_factor_ptr instead then...
            LOG(ERROR) << "The factor with slot id: " << slot_id
                        << " in the graph does not match the old_smart_factor of "
                        << "lmk with id: " << lmk_id << "\n."
                        << "Deleting old_smart_factor of lmk id: " << lmk_id;
            old_smart_factor_it = old_smart_factors_.erase(old_smart_factor_it);
            CHECK(deleteLmkFromFeatureTracks(lmk_id));
            continue;
        }
        // Why do we do this? all info is in smart_factor_ptr
        // such as the triangulated point, whether it is valid or not
        // and the number of observations...
        // Is graph more up to date?
        boost::shared_ptr<SmartStereoFactor> gsf =
            boost::dynamic_pointer_cast<SmartStereoFactor>(graph.at(slot_id));
        CHECK(gsf) << "Cannot cast factor in graph to a smart stereo factor.";
        // Get triangulation result from smart factor.
        const gtsam::TriangulationResult& result = gsf->point();
        // Check that the boost::optional result is initialized.
        // Otherwise we will be dereferencing a nullptr and we will head
        // directly to undefined behaviour wonderland.
        if (result.valid()) {
            CHECK(result.is_initialized());
            if (gsf->measured().size() >= min_age) {
                if(points_with_id.find(lmk_id) != points_with_id.end()) {
                    continue;
                }
                points_with_id[lmk_id] = *result;
                nr_valid_smart_lmks++;
            } else {
                VLOG(20) << "Rejecting lmk with id: " << lmk_id
                            << " from list of lmks in time horizon: "
                            << "not enough measurements, " << gsf->measured().size()
                            << ", vs min_age of " << min_age << ".";
            }  // gsf->measured().size() >= min_age ?
        } else {
            VLOG(20) << "Triangulation result for smart factor of lmk with id "
                    << lmk_id << " is not initialized...";
        }  // result.is_initialized()?
    }

    // Step 2: Add landmarks that now are in projection factors.
    size_t nr_proj_lmks = 0;
    // for (const gtsam::Values::Filtered<gtsam::Value>::ConstKeyValuePair&
    //     key_value : state_.filter(gtsam::Symbol::ChrTest('l'))) {
    //     DCHECK_EQ(gtsam::Symbol(key_value.key).chr(), 'l');
    //     const LandmarkId& lmk_id = gtsam::Symbol(key_value.key).index();
    //     DCHECK(points_with_id.find(lmk_id) == points_with_id.end());
    //     points_with_id[lmk_id] = key_value.value.cast<gtsam::Point3>();
    //     nr_proj_lmks++;
    // }
    // // TODO aren't these points post-optimization? Shouldn't we instead add
    // // the points before optimization? Then the regularities we enforce will
    // // have the most impact, otherwise the points in the optimization horizon
    // // do not move that much after optimizing... they are almost frozen and
    // // are not visually changing much...
    // // They might actually not be changing that much because we are not
    // // enforcing the regularities on the points that are out of current frame
    // // in the Backend currently...
    cout << "Landmark typology to be used for the mesh:\n"
            << "Number of valid smart factors " << nr_valid_smart_lmks
            << " out of " << nr_smart_lmks << "\n"
            << "Number of landmarks (not involved in a smart factor) "
            << nr_proj_lmks << ".\n Total number of landmarks: "
            << (nr_valid_smart_lmks + nr_proj_lmks);
    return points_with_id;
}
