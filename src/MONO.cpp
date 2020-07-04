#include "MONO.h"

using namespace std;

int main(int argc, char ** argv)
{
    if (argc != 2)
    {
				cerr << "Usage: ./mono_slam path_to_video" << endl;
    }

    string video_path = argv[1];
    //char image_name[200];
    cv::VideoCapture video(video_path.c_str());
    cv::Mat current_frame;
    Mono_Slam::MonoFrame Frame;

    if (!video.isOpened()){
        cerr << "Please input the right video path" << endl;
    }

    while(1) {
        video >> current_frame;

        if (current_frame.empty())
            break;

        Frame(current_frame);
    }

    /*for(int frame_i=0; frame_i <= NUM_FRAMES; frame_i++)*/
    //{
        //sprintf(image_name, (image_dir + "/%06d.png").c_str(), frame_i);
        //current_frame = cv::imread(image_name);
        //if (!current_frame.data){
            //continue;
        //}
        //Frame(current_frame);
    /*}*/


    return  0;
}

