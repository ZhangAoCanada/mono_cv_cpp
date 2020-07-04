#include "MonoFrame.h"

std::string CAMINTRIN = "../intrinsic.txt";
double s = 0.1;
int FAST_THRES = 10;

namespace Mono_Slam
{

MonoFrame::MonoFrame():scale(s), camintrinsic_filename(CAMINTRIN), 
                    fast_threshold(FAST_THRES), nonmaxSuppression(true) 
{
    // read camear intrinsic matrix
    camera_matrix = cv::Mat::zeros(cv::Size(3,3), CV_64FC1);
    readCameraIntrinsic();

    // buildng FAST detector 
    detector = cv::FastFeatureDetector::create(fast_threshold, nonmaxSuppression);

    if (!R_world.data){
        R_world = cv::Mat::eye(3, 3, CV_64FC1);
        t_world = cv::Mat::eye(3, 1, CV_64FC1);
    }

    draw = cv::Mat::zeros(TRAJECTORY_SIZE, TRAJECTORY_SIZE, CV_8UC3);
    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("sequence", cv::WINDOW_AUTOSIZE);
}

void MonoFrame::operator()(cv::InputArray image)
{
    current_frame = image.getMat().clone();
    cv::cvtColor(current_frame, current_frame_gray, cv::COLOR_BGR2GRAY);
    cv::imshow("image", current_frame);
    bool eq = false;

    if (prev_frame.data){
        eq = cv::countNonZero(current_frame_gray != prev_frame_gray) == 0;
    }

    if (!prev_frame.data || eq){
        prev_frame = current_frame.clone();
        cv::cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    }
    else{
        cv::cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
        monoFlow();
    }
    cv::waitKey(1);
}

void MonoFrame::monoFlow()
{
    featureDetection(prev_frame_gray, prev_pnts, prev_show_pnts);
    featureTracking(prev_frame_gray, current_frame_gray, prev_pnts, current_pnts, status);
    featureTracking(prev_frame_gray, current_frame_gray, prev_show_pnts, current_show_pnts, status);
    E = cv::findEssentialMat(current_pnts, prev_pnts, camera_matrix, cv::RANSAC, 0.999, 
                            1.0, mask);
    cv::recoverPose(E, current_pnts, prev_pnts, camera_matrix, R, t, mask);

    RMatToMaxAngles(R);

    // position of camera
    if ( t.at<double>(2) >= FORWARD_TRANSLATION_THRESHOLD && max_angle <= MAX_TURN_ANGLE) {
        if (prev_show_pnts.size() != 0 && current_show_pnts.size() != 0){
            pointcloudFlow();
        }
        cameraPositionFlow();
        prev_frame = current_frame.clone();
    }
    
}

void MonoFrame::cameraPositionFlow()
{
    t_world = t_world + scale * (R_world * t);
    R_world = R * R_world;

    x = int(t_world.at<double>(0)) + TRAJECTORY_SIZE/2;
    z = int(t_world.at<double>(2)) + TOP_OFFSET;

    // because the image shown with cv is upsidedown
    z = TRAJECTORY_SIZE - z;

    cv::circle(draw, cv::Point(x, z), 1, CV_RGB(255,0,0), 1);

    cv::imshow("sequence", draw);   
}

void MonoFrame::pointcloudFlow()
{
    // get projection matrix
    cv::hconcat(R, t, P2);
    cv::hconcat(cv::Mat::eye(3, 3, CV_64FC1), cv::Mat::zeros(3, 1, CV_64FC1), P1);
    
    // triangulation
    cv::Mat pnts3D(1, current_show_pnts.size(), CV_64FC4);
    cv::Mat transformed_pnts;
    //cv::triangulatePoints(P1, P2, prev_show_pnts, current_show_pnts, pnts3D);
    cv::triangulatePoints(P2, P1, prev_show_pnts, current_show_pnts, pnts3D);
    
    // world coordinate transformation
    for (int i=0; i<pnts3D.cols; i++){
        pnts3D.at<double>(0, i) = pnts3D.at<double>(0, i) / pnts3D.at<double>(3, i);
        pnts3D.at<double>(1, i) = pnts3D.at<double>(1, i) / pnts3D.at<double>(3, i);
        pnts3D.at<double>(2, i) = pnts3D.at<double>(2, i) / pnts3D.at<double>(3, i);
    }

    // convert [x, y, z, w] to [x, y, z]
    pnts3D.rowRange(0, 3).convertTo(pnts3D, CV_64FC1);
    // don't know why, but it works.
    cv::flip(pnts3D, pnts3D, 0);
    transformed_pnts = scale * (R_world * pnts3D);
    for (int i=0; i<transformed_pnts.cols; i++){
        x = int(transformed_pnts.at<double>(0, i));
        y = int(transformed_pnts.at<double>(1, i));
        z = int(transformed_pnts.at<double>(2, i));
        // remove too far and too close points
        if (std::sqrt(std::pow(x,2) + std::pow(z,2)) <= PCL_DISTANCE_UPPER &&
            std::sqrt(std::pow(x,2) + std::pow(z,2)) >= PCL_DISTANCE_LOWER &&
            pnts3D.at<double>(2, i) > 0)
        {
            x = x + int(t_world.at<double>(0)) + TRAJECTORY_SIZE/2;
            z = z + int(t_world.at<double>(2)) + TOP_OFFSET;

            // because the image shown with cv is upsidedown
            z = TRAJECTORY_SIZE - z;
            cv::circle(draw, cv::Point(x, z), 1, CV_RGB(255,255,255), 1);
        }
    }
}

void MonoFrame::RMatToMaxAngles(cv::Mat & R)
{    
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    double x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }

    max_angle = std::max(std::abs(z), std::max(std::abs(x), std::abs(y)));
}

void MonoFrame::featureTracking(cv::Mat image1, cv::Mat image2, 
        std::vector<cv::Point2f> & points1, std::vector<cv::Point2f> & points2, 
        std::vector<uchar> & status)
{
    std::vector<float> err;
    cv::Size win_size = cv::Size(21, 21);
    int unvalid_number = 0;

    cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                                30, 0.01);
    cv::calcOpticalFlowPyrLK(image1, image2, points1, points2, status, err, win_size,
                            3, termcrit, 0, 0.001);

    //remove the non matched points and negative points
    for (int i=0; i<status.size(); i++){
        cv::Point2f pnt = points2.at(i - unvalid_number);
        if ( (status.at(i) == 0) || (pnt.x < 0) || (pnt.y < 0) ){
            points1.erase(points1.begin() + (i - unvalid_number));
            points2.erase(points2.begin() + (i - unvalid_number));
            unvalid_number ++ ;
        }
    }
}

void MonoFrame::featureDetection(cv::Mat image, std::vector<cv::Point2f> & points, 
                                std::vector<cv::Point2f> & points_show)
{
    std::vector<cv::KeyPoint> keypoints;

    detector->detect(image, keypoints);

    cv::KeyPointsFilter::retainBest(keypoints, KEYPOINTS_NUM);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());

    cv::KeyPointsFilter::retainBest(keypoints, KEYPOINTS_SHOW);
    cv::KeyPoint::convert(keypoints, points_show, std::vector<int>());
}

void MonoFrame::readCameraIntrinsic()
{
    std::ifstream f;
    f.open(camintrinsic_filename.c_str());
    char value_str[100];
    std::vector<double> values;
    
    /* read camera matrix values */
    if (f.is_open()){
        while (!f.eof()){
            f >> value_str;
            values.push_back(std::stod(value_str));
        }
    }

    /* assign camear matrix values */
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++){
            camera_matrix.at<double>(i,j) = values[i*3 + j];
        }
    }
}
}

