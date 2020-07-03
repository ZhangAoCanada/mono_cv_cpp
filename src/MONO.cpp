#include "MONO.h"

using namespace std;

int main(int argc, char ** argv)
{
    if (argc != 2)
    {
				cerr << "Usage: ./mono_slam path_to_images" << endl;
    }

    string image_dir = argv[1];
    char image_name[200];
    cv::Mat current_frame;

    //cout << "mat" << current_frame << endl;
    //cout << "rows" << current_frame.rows << endl;
    //cout << "cols" << current_frame.cols << endl;

    Mono_Slam::MonoFrame Frame;

    for(int frame_i=0; frame_i <= NUM_FRAMES; frame_i++)
    {
        sprintf(image_name, (image_dir + "/%06d.png").c_str(), frame_i);
        current_frame = cv::imread(image_name);
        if (!current_frame.data){
            continue;
        }
        Frame(current_frame);
    }


    return  0;
}

