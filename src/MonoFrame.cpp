#include "MonoFrame.h"

std::string CAMINTRIN = "../intrinsic.txt";
double s = 0.5;

namespace Mono_Slam
{

MonoFrame::MonoFrame():camintrinsic_filename(CAMINTRIN), scale(s)
{
    // read camear intrinsic matrix
    camera_matrix = cv::Mat::zeros(cv::Size(3,3), CV_64FC1);
    readCameraIntrinsic();

    cv::namedWindow("sequence", cv::WINDOW_AUTOSIZE);
}

void MonoFrame::operator()(cv::InputArray image)
{
    current_frame = image.getMat();
    if (!prev_frame.data){
        prev_frame = image.getMat();
    }
    else{
        monoFlow();
        prev_frame = image.getMat();
    }
}

void MonoFrame::monoFlow()
{
    cv::Mat triangulated_pnts;

    cv::cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(current_frame, current_frame_gray, cv::COLOR_BGR2GRAY);

    featureDetection(prev_frame_gray, prev_pnts);
    featureTracking(prev_frame_gray, current_frame_gray, prev_pnts, current_pnts, status);
    E = cv::findEssentialMat(current_pnts, prev_pnts, camera_matrix, cv::RANSAC, 0.999, 1.0, mask);
    //cv::recoverPose(E, current_pnts, prev_pnts, camera_matrix, R, t, mask); 
    cv::recoverPose(E, current_pnts, prev_pnts, camera_matrix, R, t, 10, mask, triangulated_pnts); 

    std::cout << triangulated_pnts.data << std::endl;

    //if (!R_world.data){
        //R_world = R.clone();
        //t_world = t.clone();
    //} else {
        //t_world = t_world + scale * (R * t);
        //R_world = R * R_world;

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
        cv::Point2f pnt = points2[i - unvalid_number];
        if ( (status[i] == 0) || (pnt.x < 0) || (pnt.y < 0) ){
            points1.erase(points1.begin() + i - unvalid_number);
            points2.erase(points2.begin() + i - unvalid_number);
            unvalid_number ++ ;
        }
    }
}

void MonoFrame::featureDetection(cv::Mat image, std::vector<cv::Point2f> & points)
{
    std::vector<cv::KeyPoint> keypoints;
    int fast_threshold = 20;
    bool nonmaxSuppression;
    cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}

void MonoFrame::showImage(cv::Mat image)
{
    cv::imshow("sequence", image);
    cv::waitKey(10);
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

    std::cout << "camera matrix: \n" << camera_matrix << std::endl;
}

}

