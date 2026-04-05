#include "BadmintonCourtFitter.h"
#include "GlobalParameters.h"
#include "TimeMeasurement.h"
#include "DebugHelpers.h"
#include "geometry.h"

using namespace cv;
using namespace std;

bool BadmintonCourtFitter::debug = false;
const std::string BadmintonCourtFitter::windowName = "BadmintonCourtFitter";

BadmintonCourtFitter::Parameters::Parameters()
{

}

BadmintonCourtFitter::BadmintonCourtFitter()
  : BadmintonCourtFitter(Parameters())
{

}


BadmintonCourtFitter::BadmintonCourtFitter(BadmintonCourtFitter::Parameters p)
  : parameters(p)
{

}

BadmintonCourtModel BadmintonCourtFitter::run(const std::vector<Line>& lines, const Mat& binaryImage,
  const Mat& rgbImage)
{
  TimeMeasurement::start("BadmintonCourtFitter::run");

  bestScore = GlobalParameters().initialFitScore;

  TimeMeasurement::start("\tfindBestModelFit (optimization mode)");
  //findBestModelFit(lines, binaryImage, rgbImage, 1); // Original approach
  findBestModelFit2(lines, binaryImage, rgbImage);     // Our approach
  std::cerr << "Current best model score: " << bestScore << std::endl;
  TimeMeasurement::stop("\tfindBestModelFit (optimization mode)");

  // const int minThresh = 2 * 1600, goodThresh = 2 * 2100;
  // if (bestScore < minThresh) {
  //   std::cout << "Fit scores in default-mode too low. Trying more time-intensive computations..." << std::endl;
  //   TimeMeasurement::start("\tfindBestModelFit (random mode)");
  //   const int trials = 24;
  //   for (int t = 0; t < trials; t++) {
  //     //findBestModelFit(lines, binaryImage, rgbImage, 0);
  //     findBestModelFit2(lines, binaryImage, rgbImage);
  //     if (t % 5 == 0) {
  //       std::cerr << "Iteration " << t << ", current best model score: " 
  //                 << bestScore << std::endl;
  //     }
  //     if (bestScore >= goodThresh) {
  //       break;
  //     }
  //   }
  //   TimeMeasurement::stop("\tfindBestModelFit (random mode)");
  // }

  TimeMeasurement::stop("BadmintonCourtFitter::run");

  return bestModel;
}

void BadmintonCourtFitter::getHorizontalAndVerticalLines(const std::vector<Line>& lines,
  std::vector<Line>& hLines, std::vector<Line>& vLines, const cv::Mat& rgbImage, int mode)
{
  for (auto& line: lines)
  {
    if (line.isVertical())
    {
      vLines.push_back(line);
    }
    else
    {
      hLines.push_back(line);
    }
  }

  if (debug)
  {
    std::cout << "Horizontal lines = " << hLines.size() << std::endl;
    std::cout << "Vertical lines = " << vLines.size() << std::endl;
    Mat image = rgbImage.clone();
    drawLines(hLines, image, Scalar(255, 0, 0));
    drawLines(vLines, image, Scalar(0, 255, 0));
    displayImage(windowName, image);
  }
}


std::vector<LinePair> BadmintonCourtFitter::getParallelLinePairs(const std::vector<Line>& lines, const cv::Mat& rgbImage)
{
  std::vector<LinePair> linePairs;

  for (size_t i = 0; i < lines.size(); i++) 
  {
    for (size_t j = i + 1; j < lines.size(); j++) 
    {
      if (lines[i].isParallel_cp(lines[j], 0.2)) 
      {
        linePairs.push_back(std::make_pair(lines[i], lines[j]));
      }
      else 
      {
        cv::Point2f p1;
        cv::Point2f p2;
        cv::Point2f p3;

        lines[i].computeIntersectionPoint(lines[j], p1);

        int cnt = 0;
        for (size_t k = 0; k < lines.size(); k++) 
        {
          if (k == i || k == j) continue;

          lines[k].computeIntersectionPoint(lines[i], p2);
          lines[k].computeIntersectionPoint(lines[j], p3);

          float dis1 = distance(p1, p2);
          float dis2 = distance(p1, p3);

          if (dis1 <= 100 && dis2 <= 100) 
          {
            cnt += 1;
          }
        }

        if (cnt >= 1)
        {
          linePairs.push_back(std::make_pair(lines[i], lines[j]));
        }
      }

    }
  }

  if (debug)
  {
    Mat image = rgbImage.clone();
    cv::Mat resized_image;
    for (auto& lp: linePairs)
    {
      
      drawLine(lp.first, image, Scalar(0, 255, 0));
      drawLine(lp.second, image, Scalar(0, 255, 0));
      
    }
    cv::resize(image, resized_image, cv::Size(1280, 720));
    displayImage(windowName, image);
  }

  return linePairs;
}


void BadmintonCourtFitter::sortHorizontalLines(std::vector<Line>& hLines, const cv::Mat& rgbImage)
{
  auto line = Line::fromTwoPoints(Point2f(rgbImage.cols / 2, rgbImage.rows), Point2f(0, 1));
  sortLinesByLineIntersections(hLines, line);

  if (false)
  {
    for (auto& line: hLines)
    {
      Mat image = rgbImage.clone();
      drawLine(line, image, Scalar(255, 0, 0));
      displayImage(windowName, image);
    }
  }
}

void BadmintonCourtFitter::sortVerticalLines(std::vector<Line>& vLines, const cv::Mat& rgbImage)
{
  // TODO: We should be computing the hulls formed by the region below the lines
  // and picking a point in the hull, but this should be good enough.
  auto line = Line::fromTwoPoints(Point2f(0, rgbImage.rows / 2), Point2f(1, 0));
  sortLinesByLineIntersections(vLines, line);

  if (false)
  {
    for (auto& line: vLines)
    {
      Mat image = rgbImage.clone();
      drawLine(line, image, Scalar(0, 255, 0));
      displayImage(windowName, image);
    }
  }
}


void BadmintonCourtFitter::findBestModelFit(const std::vector<Line>& lines, const cv::Mat& binaryImage, const cv::Mat& rgbImage, int mode)
{
  std::vector<Line> hLines, vLines;
  getHorizontalAndVerticalLines(lines, hLines, vLines, rgbImage, mode);

  for (int flip = 0; flip < 2; flip++) {
    if (flip) {
      hLines.swap(vLines);
    }

    sortHorizontalLines(hLines, rgbImage);
    sortVerticalLines(vLines, rgbImage);

    hLinePairs = BadmintonCourtModel::getPossibleLinePairs(hLines);
    vLinePairs = BadmintonCourtModel::getPossibleLinePairs(vLines);

    if (debug)
    {
      std::cout << "Horizontal line pairs = " << hLinePairs.size() << std::endl;
      std::cout << "Vertical line pairs = " << vLinePairs.size() << std::endl;
    }

    if (hLinePairs.empty() || vLinePairs.empty())
    {
      throw std::runtime_error("Not enough line candidates were found.");
    }

    for (auto& hLinePair: hLinePairs)
    {
      for (auto& vLinePair: vLinePairs)
      {
        BadmintonCourtModel model;
        float score = model.fit(hLinePair, vLinePair, binaryImage, rgbImage);
        float netScore = 0;
        if (score > GlobalParameters().initialFitScore) {
          netScore = model.fitNet(lines, binaryImage, rgbImage);
        }
        
        // TODO: Figure out how to score the white pixels of the net
        if (score + 1e-2 * netScore > bestScore)
        {
          bestScore = score + 1e-2 * netScore;
          bestModel = model;
          std::cerr << "Score breakdown: " << score << " " << netScore << std::endl;
          if (debug) {
            Mat image = rgbImage.clone();
            drawLine(hLinePair.first, image, Scalar(255, 0, 0));
            drawLine(hLinePair.second, image, Scalar(255, 0, 0));
            drawLine(vLinePair.first, image, Scalar(255, 0, 0));
            drawLine(vLinePair.second, image, Scalar(255, 0, 0));
            displayImage(windowName, image);

            bestModel.drawModel(image);
            displayImage(windowName, image);
          }
        }

      }
    }
  }

  if (debug)
  {
    std::cout << "Best model score = " << bestScore << std::endl;
    Mat image = rgbImage.clone();
    bestModel.drawModel(image);
    displayImage(windowName, image);
  }
}

void BadmintonCourtFitter::findBestModelFit2(const std::vector<Line>& lines, const cv::Mat& binaryImage, const cv::Mat& rgbImage)
{
  hLinePairs = getParallelLinePairs(lines, rgbImage);

  std::vector<LinePair> allPairs = hLinePairs; // copy hLinePairs
  allPairs.insert(allPairs.end(), vLinePairs.begin(), vLinePairs.end()); // append vLinePairs

  if (debug)
  {
    std::cout << "H Line pairs = " << hLinePairs.size() << std::endl;
    std::cout << "V Line pairs = " << vLinePairs.size() << std::endl;
    std::cout << "Line pairs = " << allPairs.size() << std::endl;
  }

  if (allPairs.empty())
  {
    throw std::runtime_error("Not enough line candidates were found.");
  }

  for (auto& hLinePair: hLinePairs)
    {
      for (auto& vLinePair: hLinePairs)
      {

        BadmintonCourtModel model;
        float score = model.fit(hLinePair, vLinePair, binaryImage, rgbImage);
        float netScore = 0;
        if (score > GlobalParameters().initialFitScore) {
          netScore = model.fitNet(lines, binaryImage, rgbImage);
        }
        
        // TODO: Figure out how to score the white pixels of the net
        if (score + 1e-2 * netScore > bestScore)
        {
          bestScore = score + 1e-2 * netScore;
          bestModel = model;
          std::cerr << "Score breakdown: " << score << " " << netScore << std::endl;
          if (debug) {
            Mat image = rgbImage.clone();
            drawLine(hLinePair.first, image, Scalar(255, 0, 0));
            drawLine(hLinePair.second, image, Scalar(255, 0, 0));
            drawLine(vLinePair.first, image, Scalar(255, 0, 0));
            drawLine(vLinePair.second, image, Scalar(255, 0, 0));
            displayImage(windowName, image);

            bestModel.drawModel(image);
            displayImage(windowName, image);
          }
        }

      }
    }
}

