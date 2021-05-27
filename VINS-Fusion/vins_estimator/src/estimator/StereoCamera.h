#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>


#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Pose3.h>

class CameraParams {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // fu, fv, cu, cv
  using Distortion = std::vector<double>;
  CameraParams(
    const cv::Size image_size,
    const gtsam::Pose3 &body_Pose_cam,
    const std::array<double, 4> &intrinsics,
    const std::vector<double> &distortion_coeff) : 
        image_size_(image_size),
        body_Pose_cam_(body_Pose_cam),
        intrinsics_(intrinsics),
        distortion_coeff_(distortion_coeff) {
    convertIntrinsicsVectorToMatrix(intrinsics_, &K_);
    convertDistortionVectorToMatrix(distortion_coeff_, &distortion_coeff_mat_);
  }
  ~CameraParams() = default;
  void convertDistortionVectorToMatrix(
      const std::vector<double>& distortion_coeffs,
      cv::Mat* distortion_coeffs_mat) {
    // CHECK_NOTNULL(distortion_coeffs_mat);
    // CHECK_GE(distortion_coeffs.size(), 4u);
    *distortion_coeffs_mat = cv::Mat::zeros(1, distortion_coeffs.size(), CV_64F);
    for (int k = 0; k < distortion_coeffs_mat->cols; k++) {
      distortion_coeffs_mat->at<double>(0, k) = distortion_coeffs[k];
    }
  }
  void convertIntrinsicsVectorToMatrix(const std::array<double, 4>& intrinsics,
                                                    cv::Mat* camera_matrix) {
    // CHECK_NOTNULL(camera_matrix);
    // DCHECK_EQ(intrinsics.size(), 4);
    *camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix->at<double>(0, 0) = intrinsics[0];
    camera_matrix->at<double>(1, 1) = intrinsics[1];
    camera_matrix->at<double>(0, 2) = intrinsics[2];
    camera_matrix->at<double>(1, 2) = intrinsics[3];
  }

 public:
  //! Image info.
  // double frame_rate_;
  cv::Size image_size_;
  //! Sensor extrinsics wrt body-frame
  gtsam::Pose3 body_Pose_cam_;
  //! fu, fv, cu, cv
  std::array<double, 4> intrinsics_;
  //! OpenCV structures: needed to compute the undistortion map.
  //! 3x3 camera matrix K (last row is {0,0,1})
  cv::Mat K_;
  //! Distortion parameters
  std::vector<double> distortion_coeff_;
  cv::Mat distortion_coeff_mat_;

};

class StereoCamera {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Baseline = double;
  StereoCamera(const CameraParams& left_cam_params,
               const CameraParams& right_cam_params);
  virtual ~StereoCamera() = default;

 public:

  /**
   * @brief getBodyPoseLeftCamRect Get left camera pose after rectification with
   * respect to the body frame.
   * @return
   */
  inline gtsam::Pose3 getBodyPoseLeftCamRect() const {
    return B_Pose_camLrect_;
  }
  /**
   * @brief getBodyPoseRightCamRect Get right camera pose after rectification
   * with respect to the body frame.
   * @return
   */
  inline gtsam::Pose3 getBodyPoseRightCamRect() const {
    return B_Pose_camRrect_;
  }

  // Ideally this would return a const shared pointer or a copy, but GTSAM's
  // idiosyncrasies require shared ptrs all over the place.
  /**
   * @brief getStereoCalib
   * @return stereo camera calibration after undistortion and rectification.
   */
  inline gtsam::Cal3_S2Stereo::shared_ptr getStereoCalib() const {
    if (stereo_calibration_ != nullptr){
      return stereo_calibration_;
    } else {
      printf("stereo_calibration_ is nullptr! \n");
      assert(0);
    }
  }

  // void rectifiedPoint2f(const cv::Point2f p, cv::Point2f *rec_p, const CameraParams &camera_params) {
  //   cv::undistortPoints(p, *rec_p, camera_params.K_, camera_params.distortion_coeff_mat_, camera_params.K_);
  // }

  /**
   * @brief getImageSize
   * @return image size of left/right frames
   */
  inline cv::Size getImageSize() const {
    // CHECK_EQ(ROI1_, ROI2_);
    return cv::Size(ROI1_.x, ROI1_.y);
  }
  inline cv::Rect getROI1() const { return ROI1_; }
  inline cv::Rect getROI2() const { return ROI2_; }

  inline cv::Mat getP1() const { return P1_; }

  inline cv::Mat getP2() const { return P2_; }

  inline cv::Mat getR1() const { return R1_; }

  inline cv::Mat getR2() const { return R2_; }

  inline cv::Mat getQ() const { return Q_; }

  inline Baseline getBaseline() const { return stereo_baseline_; }
  /**
   * @brief computeRectificationParameters
   *
   * Outputs new rotation matrices R1,R2
   * so that the image planes of the stereo camera are parallel.
   * It also outputs new projection matrices P1, P2, and a disparity to depth
   * matrix for stereo pointcloud reconstruction.
   *
   * @param left_cam_params Left camera parameters
   * @param right_cam_params Right camera parameters
   *
   * @param R1 Output 3x3 rectification transform (rotation matrix) for the
   * first camera.
   * @param R2 Output 3x3 rectification transform (rotation matrix) for the
   * second camera.
   * @param P1 Output 3x4 projection matrix in the new (rectified) coordinate
   * systems for the first camera.
   * @param P2 Output 3x4 projection matrix in the new (rectified) coordinate
   * systems for the second camera.
   * @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see
   * reprojectImageTo3D ).
   * @param ROI1 Region of interest in image 1.
   * @param ROI2 Region of interest in image 2.
   */
  static void computeRectificationParameters(
      const CameraParams& left_cam_params,
      const CameraParams& right_cam_params,
      cv::Mat* R1,
      cv::Mat* R2,
      cv::Mat* P1,
      cv::Mat* P2,
      cv::Mat* Q,
      cv::Rect* ROI1,
      cv::Rect* ROI2);

 public:
  gtsam::Pose3 camL_Pose_camLrect_;
  gtsam::Pose3 camR_Pose_camRrect_;
 private:
  //! Stereo camera calibration
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;
  //! Pose from Body to Left/Right Camera after rectification
  gtsam::Pose3 B_Pose_camLrect_;
  gtsam::Pose3 B_Pose_camRrect_;
  // TODO(Toni): perhaps wrap these params in a struct instead.
  /// Projection matrices after rectification
  /// P1,P2 Output 3x4 projection matrix in the new (rectified) coordinate
  /// systems for the left and right camera (see cv::stereoRectify).
  cv::Mat P1_, P2_;
  /// R1,R2 Output 3x3 rectification transform (rotation matrix) for the left
  /// and for the right camera.
  cv::Mat R1_, R2_;
  /// Q Output 4x4 disparity-to-depth mapping matrix (see
  /// cv::reprojectImageTo3D or cv::stereoRectify).
  cv::Mat Q_;
  //! Regions of interest in the left/right image.
  cv::Rect ROI1_, ROI2_;
  //! Stereo baseline
  Baseline stereo_baseline_;

  // //! Left and right camera objects.
  // //! These are neither undistorted nor rectified
  // VIO::Camera::ConstPtr original_left_camera_;
  // VIO::Camera::ConstPtr original_right_camera_;
  // //! Stereo camera implementation
  // gtsam::StereoCamera undistorted_rectified_stereo_camera_impl_;
  // //! Undistortion rectification pre-computed maps for cv::remap
  // UndistorterRectifier::UniquePtr left_cam_undistort_rectifier_;
  // UndistorterRectifier::UniquePtr right_cam_undistort_rectifier_;

};
