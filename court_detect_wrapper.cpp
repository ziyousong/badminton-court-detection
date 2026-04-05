#include <opencv2/opencv.hpp>
#include "CourtLinePixelDetector.h"
#include "CourtLineCandidateDetector.h"
#include "BadmintonCourtFitter.h"

extern "C" {

    void run_court_detect(CourtLinePixelDetector::Parameters p, unsigned char* data, int width, int height, int channels, unsigned char* output, cv::Point2f* points) {
        if (!data || !output) return;

        // Convert raw image data into OpenCV Mat
        cv::Mat inputImage(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, data);

        CourtLinePixelDetector courtLinePixelDetector(p);
        CourtLineCandidateDetector courtLineCandidateDetector;
        BadmintonCourtFitter badmintonCourtFitter;

        cv::Mat binaryImage = courtLinePixelDetector.run(inputImage);
        std::vector<Line> candidateLines = courtLineCandidateDetector.run(binaryImage, inputImage);
        BadmintonCourtModel model = badmintonCourtFitter.run(candidateLines, binaryImage, inputImage);

        model.drawModel(inputImage);

        memcpy(output, inputImage.data, width * height * channels);

        model.getPoints(points);
    }


}
