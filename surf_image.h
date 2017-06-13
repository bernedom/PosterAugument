#ifndef SURF_IMAGE_H
#define SURF_IMAGE_H

#include <opencv2/core/core.hpp>
#include <vector>

struct SURF_Image {
  // cannot use std::array despite fixed size because opencv doesn't use it yet
  std::vector<cv::Point2f> corners{cv::Point2f(), cv::Point2f(), cv::Point2f(),
                                   cv::Point2f()};
  std::vector<cv::Point2f> coords;
  cv::Mat raw_data;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

#endif // SURF_IMAGE_H
