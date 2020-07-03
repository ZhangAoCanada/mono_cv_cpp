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
#include <fstream>
#include <sstream>
#include <string>

namespace Mono_Slam
{
class MonoFrame
{
public:
    MonoFrame();
    void showImage(cv::Mat image);
    void operator()(cv::InputArray image);

protected:
    void readCameraIntrinsic();
    void monoFlow();
    void featureDetection(cv::Mat image, std::vector<cv::Point2f> & points);
    void featureTracking(cv::Mat image1, cv::Mat image2, std::vector<cv::Point2f> & points1, 
                        std::vector<cv::Point2f> & points2, std::vector<uchar> & status);

    std::vector<cv::Point2f> prev_pnts;
    std::vector<cv::Point2f> current_pnts;
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

    cv::Mat R_world;
    cv::Mat t_world;
    
};
}

#endif
