#ifndef MONOFRAME_H
#define MONOFRAME_H

#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

namespace Mono_Slam
{

#define TRAJECTORY_SIZE 600
#define TOP_OFFSET 100
#define KEYPOINTS_NUM 3000
#define MAX_TURN_ANGLE 3.141593/2
#define FORWARD_TRANSLATION_THRESHOLD 0.1

#define KEYPOINTS_SHOW 10
#define PCL_DISTANCE_UPPER 1000
#define PCL_DISTANCE_LOWER 10

class MonoFrame
{
public:
    MonoFrame();
    void operator()(cv::InputArray image);

protected:
    void readCameraIntrinsic();
    void monoFlow();
    void pointcloudFlow();
    void cameraPositionFlow();
    void RMatToMaxAngles(cv::Mat & R);
    void featureDetection(cv::Mat image, std::vector<cv::Point2f> & points, 
                        std::vector<cv::Point2f> & points_show);
    void featureTracking(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2f> & points1, 
                        std::vector<cv::Point2f> & points2, std::vector<uchar> & status);

    std::vector<cv::Point2f> prev_pnts;
    std::vector<cv::Point2f> current_pnts;
    std::vector<cv::Point2f> prev_show_pnts;
    std::vector<cv::Point2f> current_show_pnts;
    std::vector<uchar> status;

    cv::Mat prev_frame;
    cv::Mat current_frame;
    cv::Mat prev_frame_gray;
    cv::Mat current_frame_gray;

    cv::Mat camera_matrix;
    std::string camintrinsic_filename;
    double scale;

    cv::Mat E;
    cv::Mat R;
    cv::Mat t;
    cv::Mat mask;
    double max_angle;

    cv::Mat P1;
    cv::Mat P2;

    int fast_threshold;
    bool nonmaxSuppression;
    cv::Ptr<cv::FastFeatureDetector> detector;

    cv::Mat R_world;
    cv::Mat t_world;
    int x;
    int y;
    int z;

    cv::Mat draw;

};
}

#endif
