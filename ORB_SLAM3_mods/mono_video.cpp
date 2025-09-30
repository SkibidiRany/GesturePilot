/**
 * ORB-SLAM3 mono_video: play video file or webcam stream (real or virtual like OBS)
 */

#include "global.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <System.h>

#ifdef _WIN32
#include <windows.h>
static inline void usleep(unsigned long usec) { Sleep((DWORD)(usec / 1000)); }
#endif

using namespace std;

int mono_video(int argc, char** argv)
{
    if (argc != 4)
    {
        cerr << endl << "Usage: ./slam.exe mono_video path_to_vocabulary path_to_settings path_to_video_or_camera" << endl;
        cerr << "       Example with file:   ./slam.exe mono_video ORBvoc.txt TUM1.yaml video.mp4" << endl;
        cerr << "       Example with webcam: ./slam.exe mono_video ORBvoc.txt TUM1.yaml 0" << endl;
        return 1;
    }

    const string vocabFile = string(argv[1]);
    const string settingsFile = string(argv[2]);
    const string inputArg = string(argv[3]);

    cout << endl << "-------" << endl;
    cout << "ORB-SLAM3 mono_video " << vocabFile << " " << settingsFile << " " << inputArg << endl;

    cv::VideoCapture cap;
    bool opened = false;

    // check if input is a number (webcam index)
    if (inputArg.size() == 1 && isdigit(inputArg[0])) {
        int camIndex = stoi(inputArg);
        cout << "Trying to open webcam index " << camIndex << " ..." << endl;

#ifdef _WIN32
        // try DirectShow first (needed for OBS VirtualCam usually)
        cout << "Trying DirectShow backend (CAP_DSHOW)..." << endl;
        if (cap.open(camIndex, cv::CAP_DSHOW)) {
            cout << "Opened webcam index " << camIndex << " with CAP_DSHOW" << endl;
            opened = true;
        }
#endif
        if (!opened) {
            cout << "Trying CAP_ANY backend..." << endl;
            if (cap.open(camIndex, cv::CAP_ANY)) {
                cout << "Opened webcam index " << camIndex << " with CAP_ANY" << endl;
                opened = true;
            }
        }
    }

    // otherwise, treat as video file path
    if (!opened) {
        struct BackendTry { int id; const char* name; };
        BackendTry backends[] = {
#ifdef CV_CAP_FFMPEG
            {cv::CAP_FFMPEG, "CAP_FFMPEG"},
#endif
#ifdef CV_CAP_MSMF
            {cv::CAP_MSMF,   "CAP_MSMF"},
#endif
#ifdef CV_CAP_DSHOW
            {cv::CAP_DSHOW,  "CAP_DSHOW"},
#endif
            {cv::CAP_ANY,    "CAP_ANY"}
        };

        for (const auto& b : backends) {
            try {
                cout << "Trying to open file with backend: " << b.name << endl;
                if (cap.open(inputArg, b.id)) {
                    cout << "Opened video with backend: " << b.name << endl;
                    opened = true;
                    break;
                }
                else {
                    cout << "Failed to open with backend: " << b.name << endl;
                }
            }
            catch (const cv::Exception& e) {
                cout << "Exception while opening with " << b.name << ": " << e.what() << endl;
            }
        }
    }

    if (!opened) {
        cerr << "Failed to open input: " << inputArg << endl;
        cerr << "Suggestions:" << endl;
        cerr << "  - For webcam use '0'" << endl;
        cerr << "  - For OBS virtual cam: make sure OBS VirtualCam plugin is running, and try CAP_DSHOW backend" << endl;
        cerr << "  - For video: ensure OpenCV ffmpeg DLL is available, or re-encode video" << endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0 || fps > 120.0) fps = 30.0; // fallback if fps invalid
    double T = 1.0 / fps;

    cout << "Input: " << inputArg << endl;
    cout << "FPS (reported): " << fps << "   frame interval T = " << T << " s" << endl << endl;

    // create SLAM system
    ORB_SLAM3::System SLAM(vocabFile, settingsFile, ORB_SLAM3::System::MONOCULAR, true);

    cv::Mat im;
    vector<float> vTimesTrack;

    while (true)
    {
        if (!cap.read(im)) {
            cout << "No more frames or failed to grab frame." << endl;
            break;
        }

        double tframe;
        if (inputArg.size() == 1 && isdigit(inputArg[0])) {
            // Webcam or virtual cam: use system clock
            tframe = chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now().time_since_epoch()).count();
        }
        else {
            // Video file: use video timestamp
            tframe = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        }

        auto t1 = chrono::steady_clock::now();

        SLAM.TrackMonocular(im, tframe);

        auto t2 = chrono::steady_clock::now();
        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack.push_back((float)ttrack);

        cv::imshow("Frame", im);
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC

        if (ttrack < T) usleep((unsigned long)((T - ttrack) * 1e6));
    }

    SLAM.Shutdown();

    if (!vTimesTrack.empty())
    {
        sort(vTimesTrack.begin(), vTimesTrack.end());
        double totaltime = 0;
        for (auto t : vTimesTrack) totaltime += t;

        cout << "-------" << endl;
        cout << "Frames processed: " << vTimesTrack.size() << endl;
        cout << "median tracking time: " << vTimesTrack[vTimesTrack.size() / 2] << endl;
        cout << "mean tracking time: " << totaltime / vTimesTrack.size() << endl;
    }

#ifdef _WIN32
    system("if not exist log mkdir log");
#else
    system("mkdir -p log");
#endif

    SLAM.SaveKeyFrameTrajectoryTUM("log/KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("log/CameraTrajectory.txt");

    cout << "Finished. Trajectories saved in log/ folder." << endl;
    return 0;
}
