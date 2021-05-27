// #include <stdlib.h>
// #include <sys/time.h>
// #include <fstream>
// #include <iomanip>
// #include <sstream>

// #include <opengv/point_cloud/methods.hpp>


// #include <gtsam/base/Matrix.h>
// #include <gtsam/base/Vector.h>
// #include <gtsam/geometry/Cal3DS2.h>
// #include <gtsam/geometry/Cal3_S2.h>
// #include <gtsam/geometry/Point2.h>
// #include <gtsam/geometry/Point3.h>
// #include <gtsam/geometry/Pose3.h>
// #include <gtsam/navigation/ImuBias.h>


#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <boost/utility.hpp>  // for tie
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoCamera.h>

#include "stereo_camera.h"
using namespace std;
// Converts a 3x3 rotation matrix from opencv to gtsam Rot3
gtsam::Rot3 cvMatToGtsamRot3(const cv::Mat& R) {
//   CHECK_EQ(R.rows, 3);
//   CHECK_EQ(R.cols, 3);
  gtsam::Matrix rot_mat = gtsam::Matrix::Identity(3, 3);
  cv::cv2eigen(R, rot_mat);
  return gtsam::Rot3(rot_mat);
}

// Converts a 3x3 (or 3xM, M > 3) camera matrix from opencv to gtsam::Cal3_S2.
gtsam::Cal3_S2 Cvmat2Cal3_S2(const cv::Mat& M) {
//   CHECK_EQ(M.rows, 3);  // We expect homogeneous camera matrix.
//   CHECK_GE(M.cols, 3);  // We accept extra columns (which we do not use).
  const double& fx = M.at<double>(0, 0);
  const double& fy = M.at<double>(1, 1);
  const double& s = M.at<double>(0, 1);
  const double& u0 = M.at<double>(0, 2);
  const double& v0 = M.at<double>(1, 2);
  return gtsam::Cal3_S2(fx, fy, s, u0, v0);
}

// Converts a gtsam pose3 to a 3x3 rotation matrix and translation vector
// in opencv format (note: the function only extracts R and t, without changing
// them)
std::pair<cv::Mat, cv::Mat> Pose2cvmats(const gtsam::Pose3& pose) {
  const gtsam::Matrix3& rot = pose.rotation().matrix();
  const gtsam::Vector3& tran = pose.translation();
  cv::Mat R = cv::Mat(3, 3, CV_64F);
  cv::eigen2cv(rot, R);
  cv::Mat T = cv::Mat(3, 1, CV_64F);
  cv::eigen2cv(tran, T);
  return std::make_pair(R, T);
}



// StereoCamera::StereoCamera(const CameraParams& left_cam_params,
//                            const CameraParams& right_cam_params)
//     : original_left_camera_(nullptr),
//       original_right_camera_(nullptr),
//       undistorted_rectified_stereo_camera_impl_(),
//       stereo_calibration_(nullptr),
//       stereo_baseline_(0.0),
//       left_cam_undistort_rectifier_(nullptr),
//       right_cam_undistort_rectifier_(nullptr) {

StereoCamera::StereoCamera(const CameraParams& left_cam_params,
                           const CameraParams& right_cam_params)
    : stereo_calibration_(nullptr),
      stereo_baseline_(0.0){
  computeRectificationParameters(left_cam_params,
                                 right_cam_params,
                                 &R1_,
                                 &R2_,
                                 &P1_,
                                 &P2_,
                                 &Q_,
                                 &ROI1_,
                                 &ROI2_);
  // Calc left camera pose after rectification
  // NOTE: OpenCV pose convention is the opposite, therefore the inverse.

  printf("-----------------\n");
  cout<<"left_cam_params: "<<endl<<left_cam_params.K_<<endl;
  cout<<"right_cam_params: "<<endl<<right_cam_params.K_<<endl;
  cout<<"R1_: "<<endl<<R1_<<endl;
  cout<<"R2_: "<<endl<<R2_<<endl;
  cout<<"P1_: "<<endl<<P1_<<endl;
  cout<<"P2_: "<<endl<<P2_<<endl;
  cout<<"Q_: "<<endl<<Q_<<endl;
  printf("-----------------\n");
  const gtsam::Rot3& camL_Rot_camLrect = cvMatToGtsamRot3(R1_).inverse();
  camL_Pose_camLrect_ = gtsam::Pose3(camL_Rot_camLrect, gtsam::Point3::Zero());
  B_Pose_camLrect_ = left_cam_params.body_Pose_cam_.compose(camL_Pose_camLrect_);

  const gtsam::Rot3& camR_Rot_camRrect = cvMatToGtsamRot3(R2_).inverse();
  camR_Pose_camRrect_ = gtsam::Pose3(camR_Rot_camRrect, gtsam::Point3::Zero());
  B_Pose_camRrect_ = right_cam_params.body_Pose_cam_.compose(camR_Pose_camRrect_);

  // Calc baseline (see L.2700 and L.2616 in
  // https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibration.cpp
  // NOTE: OpenCV pose convention is the opposite, therefore the missing -1.0
//   CHECK_NE(Q_.at<double>(3, 2), 0.0);
  stereo_baseline_ = 1.0 / Q_.at<double>(3, 2);
//   CHECK_GT(stereo_baseline_, 0.0);

  //! Create stereo camera calibration after rectification and undistortion.
  const gtsam::Cal3_S2& left_undist_rect_cam_mat = Cvmat2Cal3_S2(P1_);
  stereo_calibration_ =
      boost::make_shared<gtsam::Cal3_S2Stereo>(left_undist_rect_cam_mat.fx(),
                                               left_undist_rect_cam_mat.fy(),
                                               left_undist_rect_cam_mat.skew(),
                                               left_undist_rect_cam_mat.px(),
                                               left_undist_rect_cam_mat.py(),
                                               stereo_baseline_);
  // what we need : 
  // B_Pose_camLrect_
  // stereo_calibration_
  // left_cam_undistort_rectifier_ =
  //     VIO::make_unique<UndistorterRectifier>(P1_, left_cam_params, R1_);
  // right_cam_undistort_rectifier_ =
  //     VIO::make_unique<UndistorterRectifier>(P2_, right_cam_params, R2_);

  // //! Create stereo camera implementation
  // undistorted_rectified_stereo_camera_impl_ =
  //     gtsam::StereoCamera(B_Pose_camLrect_, stereo_calibration_);

}

void StereoCamera::computeRectificationParameters(
    const CameraParams& left_cam_params,
    const CameraParams& right_cam_params,
    cv::Mat* R1,
    cv::Mat* R2,
    cv::Mat* P1,
    cv::Mat* P2,
    cv::Mat* Q,
    cv::Rect* ROI1,
    cv::Rect* ROI2) {
    //   CHECK_NOTNULL(R1);
    //   CHECK_NOTNULL(R2);
    //   CHECK_NOTNULL(P1);
    //   CHECK_NOTNULL(P2);
    //   CHECK_NOTNULL(Q);
    //   CHECK_NOTNULL(ROI1);
    //   CHECK_NOTNULL(ROI2);
  //! Extrinsics of the stereo (not rectified) relative pose between cameras
  gtsam::Pose3 camL_Pose_camR =
      (left_cam_params.body_Pose_cam_).between(right_cam_params.body_Pose_cam_);
  // Get extrinsics in open CV format.
  // NOTE: openCV pose convention is the opposite, that's why we have to
  // invert
  cv::Mat camL_Rot_camR, camL_Tran_camR;
  boost::tie(camL_Rot_camR, camL_Tran_camR) =
      Pose2cvmats(camL_Pose_camR.inverse());
  // kAlpha is -1 by default, but that introduces invalid keypoints!
  // here we should use kAlpha = 0 so we get only valid pixels...
  // But that has an issue that it removes large part of the image, check:
  // https://github.com/opencv/opencv/issues/7240 for this issue with kAlpha
  // Setting to -1 to make it easy, but it should NOT be -1!
  static constexpr int kAlpha = 0;
  cv::stereoRectify(
      // Input
      left_cam_params.K_,
      left_cam_params.distortion_coeff_mat_,
      right_cam_params.K_,
      right_cam_params.distortion_coeff_mat_,
      left_cam_params.image_size_,
      camL_Rot_camR,
      camL_Tran_camR,
      // Output
      *R1,
      *R2,
      *P1,
      *P2,
      *Q,
      cv::CALIB_ZERO_DISPARITY,
      kAlpha,
      cv::Size(),
      ROI1,
      ROI2);
}
