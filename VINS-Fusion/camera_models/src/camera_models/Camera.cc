#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include <opencv2/calib3d/calib3d.hpp>

namespace camodocal
{

Camera::Parameters::Parameters(ModelType modelType)
 : m_modelType(modelType)
 , m_imageWidth(0)
 , m_imageHeight(0)
{
    switch (modelType)
    {
    case KANNALA_BRANDT:
        m_nIntrinsics = 8;
        break;
    case PINHOLE:
        m_nIntrinsics = 8;
        break;
    case SCARAMUZZA:
        m_nIntrinsics = SCARAMUZZA_CAMERA_NUM_PARAMS;
        break;
    case MEI:
    default:
        m_nIntrinsics = 9;
    }
}

Camera::Parameters::Parameters(ModelType modelType,
                               const std::string& cameraName,
                               int w, int h)
 : m_modelType(modelType)
 , m_cameraName(cameraName)
 , m_imageWidth(w)
 , m_imageHeight(h)
{
    switch (modelType)
    {
    // Kannala-Brandt Camera Model 号称是一个 generic camera model,适用于
    // 普通、广角和鱼眼镜头
    // KB 模型假设图像光心到投影点的距离和角度的多项式存在比例关系，这个角度是点所在的投影光线和主轴之间的夹角
    // KB 模型有时候被作为真空相机的畸变模型，同时也是opencv中使用的鱼眼相机模型，而在kalibr中，被用作等距畸变模型(equidistant distortion model)
    case KANNALA_BRANDT:
        m_nIntrinsics = 8;
        break;
    case PINHOLE:
        m_nIntrinsics = 8;
        break;
    case SCARAMUZZA:
        m_nIntrinsics = SCARAMUZZA_CAMERA_NUM_PARAMS;
        break;
    // MEI : 该模型2007年提出来的，由于 Kannala-Brandt 模型所提出的多项式尽管能建模大部分的投影模型，
    // 但是扔遵循针孔成像的基本假设，导致该模型存在奇点（水平入射的光线将被投影到无穷远）。MEI 因此被提出当做
    // 一个更通用的投影模型， MEI 模型将投影成像过程分为四步：
        // 1. 相机空间坐标系 Cm 下的点投影到单位球面
        // 2. 求取所投影的单位球面点在偏移坐标系Cp下的坐标
        // 3. 然后求取该点在Cp下的归一化平面坐标
        // 4. 最后再将内参数矩阵作用于该归一化平面得到像平面坐标
        // 可以看出，进过Cp下表达坐标点，可以投影更大范围的点，使得此模型对于鱼眼镜头的适用范围更广
    case MEI:
    default:
        m_nIntrinsics = 9;
    }
}

Camera::ModelType&
Camera::Parameters::modelType(void)
{
    return m_modelType;
}

std::string&
Camera::Parameters::cameraName(void)
{
    return m_cameraName;
}

int&
Camera::Parameters::imageWidth(void)
{
    return m_imageWidth;
}

int&
Camera::Parameters::imageHeight(void)
{
    return m_imageHeight;
}

Camera::ModelType
Camera::Parameters::modelType(void) const
{
    return m_modelType;
}

const std::string&
Camera::Parameters::cameraName(void) const
{
    return m_cameraName;
}

int
Camera::Parameters::imageWidth(void) const
{
    return m_imageWidth;
}

int
Camera::Parameters::imageHeight(void) const
{
    return m_imageHeight;
}

int
Camera::Parameters::nIntrinsics(void) const
{
    return m_nIntrinsics;
}

cv::Mat&
Camera::mask(void)
{
    return m_mask;
}

const cv::Mat&
Camera::mask(void) const
{
    return m_mask;
}

void
Camera::estimateExtrinsics(const std::vector<cv::Point3f>& objectPoints,
                           const std::vector<cv::Point2f>& imagePoints,
                           cv::Mat& rvec, cv::Mat& tvec) const
{
    std::vector<cv::Point2f> Ms(imagePoints.size());
    for (size_t i = 0; i < Ms.size(); ++i)
    {
        Eigen::Vector3d P;
        liftProjective(Eigen::Vector2d(imagePoints.at(i).x, imagePoints.at(i).y), P);

        P /= P(2);

        Ms.at(i).x = P(0);
        Ms.at(i).y = P(1);
    }

    // assume unit focal length, zero principal point, and zero distortion
    cv::solvePnP(objectPoints, Ms, cv::Mat::eye(3, 3, CV_64F), cv::noArray(), rvec, tvec);
}

double
Camera::reprojectionDist(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2) const
{
    Eigen::Vector2d p1, p2;

    spaceToPlane(P1, p1);
    spaceToPlane(P2, p2);

    return (p1 - p2).norm();
}

double
Camera::reprojectionError(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                          const std::vector< std::vector<cv::Point2f> >& imagePoints,
                          const std::vector<cv::Mat>& rvecs,
                          const std::vector<cv::Mat>& tvecs,
                          cv::OutputArray _perViewErrors) const
{
    int imageCount = objectPoints.size();
    size_t pointsSoFar = 0;
    double totalErr = 0.0;

    bool computePerViewErrors = _perViewErrors.needed();
    cv::Mat perViewErrors;
    if (computePerViewErrors)
    {
        _perViewErrors.create(imageCount, 1, CV_64F);
        perViewErrors = _perViewErrors.getMat();
    }

    for (int i = 0; i < imageCount; ++i)
    {
        size_t pointCount = imagePoints.at(i).size();

        pointsSoFar += pointCount;

        std::vector<cv::Point2f> estImagePoints;
        projectPoints(objectPoints.at(i), rvecs.at(i), tvecs.at(i),
                      estImagePoints);

        double err = 0.0;
        for (size_t j = 0; j < imagePoints.at(i).size(); ++j)
        {
            err += cv::norm(imagePoints.at(i).at(j) - estImagePoints.at(j));
        }

        if (computePerViewErrors)
        {
            perViewErrors.at<double>(i) = err / pointCount;
        }

        totalErr += err;
    }

    return totalErr / pointsSoFar;
}

double
Camera::reprojectionError(const Eigen::Vector3d& P,
                          const Eigen::Quaterniond& camera_q,
                          const Eigen::Vector3d& camera_t,
                          const Eigen::Vector2d& observed_p) const
{
    Eigen::Vector3d P_cam = camera_q.toRotationMatrix() * P + camera_t;

    Eigen::Vector2d p;
    spaceToPlane(P_cam, p);

    return (p - observed_p).norm();
}

void
Camera::projectPoints(const std::vector<cv::Point3f>& objectPoints,
                      const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      std::vector<cv::Point2f>& imagePoints) const
{
    // project 3D object points to the image plane
    imagePoints.reserve(objectPoints.size());

    //double
    cv::Mat R0;
    cv::Rodrigues(rvec, R0);

    Eigen::MatrixXd R(3,3);
    R << R0.at<double>(0,0), R0.at<double>(0,1), R0.at<double>(0,2),
         R0.at<double>(1,0), R0.at<double>(1,1), R0.at<double>(1,2),
         R0.at<double>(2,0), R0.at<double>(2,1), R0.at<double>(2,2);

    Eigen::Vector3d t;
    t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
        const cv::Point3f& objectPoint = objectPoints.at(i);

        // Rotate and translate
        Eigen::Vector3d P;
        P << objectPoint.x, objectPoint.y, objectPoint.z;

        P = R * P + t;

        Eigen::Vector2d p;
        spaceToPlane(P, p);

        imagePoints.push_back(cv::Point2f(p(0), p(1)));
    }
}

}
