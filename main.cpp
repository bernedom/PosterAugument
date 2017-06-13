#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp >

#include <algorithm>
#include <iostream>
#include <vector>

int main(int, char **) {

  cv::VideoCapture stream(0); // no const, because read() is not const
  if (!stream.isOpened()) {
    std::cerr << "Could not open camera\n";
    return -1;
  }

  const auto source_image =
      cv::imread("C:\\Code\\PosterAugument\\Assets\\JollyRoger.jpg");

  // cannot use std::array despite fixed size because opencv doesn't use it yet
  const std::vector<cv::Point2f> source_corners{
      cv::Point2f(0, 0), cv::Point2f((float)source_image.cols, 0),
      cv::Point2f((float)source_image.cols, (float)source_image.rows),
      cv::Point2f(0, (float)source_image.rows)};

  const auto min_hessian = 400;
  cv::Ptr<cv::xfeatures2d::SURF> detector =
      cv::xfeatures2d::SURF::create(min_hessian);

  std::vector<cv::KeyPoint> keypoints_source;
  cv::Mat descriptors_source;

  detector->detect(source_image, keypoints_source);
  detector->compute(source_image, keypoints_source, descriptors_source);

  cv::Mat debug_img_source;
  cv::drawKeypoints(source_image, keypoints_source, debug_img_source);
  cv::imshow("source", debug_img_source); // debug drawing of keypoints

  std::vector<cv::KeyPoint> keypoints_videoframe;
  cv::Mat debug_img_videoframe;

  while (true) {
    keypoints_videoframe.clear();

    cv::Mat videoframe;
    cv::Mat descriptors_videoframe;

    stream.read(videoframe); // slurp a single frame from the webcam
    detector->detect(videoframe, keypoints_videoframe);
    detector->compute(videoframe, keypoints_videoframe, descriptors_videoframe);

    if (descriptors_videoframe.cols == 0 || descriptors_videoframe.rows == 0) {
      std::cout << "Not enough features to perform matching, skipping frame\n";
      continue;
    }

    // find matches between the keypoints
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_source, descriptors_videoframe, matches);

    // get the min & max distance (max is not
    const auto cmp_func = [](const cv::DMatch &lhs, const cv::DMatch &rhs) {
      return lhs.distance < rhs.distance;
    };
    // this will frequently yield 0 which is a good indicator for matching
    // points
    const auto min_distance =
        std::min_element(matches.begin(), matches.end(), cmp_func);

    std::vector<cv::Point2f> videoframe_coords;
    std::vector<cv::Point2f> source_coords;
    // this culling using a multiplier is to better find fuzzy matches (having
    // min_dist > 0)
    const double max_distance_multiplier = 3.0;
    std::vector<cv::DMatch> culled_matches;
    for (const auto &m : matches) {
      if (m.distance < min_distance->distance * max_distance_multiplier) {
        culled_matches.push_back(m);
        source_coords.push_back(keypoints_source[m.queryIdx].pt);
        videoframe_coords.push_back(keypoints_videoframe[m.trainIdx].pt);
      }
    }

    std::cout << culled_matches.size() << std::endl;

    // creating a set of homogenous transforms between points to do pose
    // estimation, limiting the number of RANSAC iterations so it performs a bit
    // faster
    cv::Mat homography = cv::findHomography(source_coords, videoframe_coords,
                                            CV_RANSAC, 3, cv::noArray(), 1000);
    std::vector<cv::Point2f> videoframe_corners(4);
    // this might fail for no aparent reason because RANSAC might not find a
    // consensus given the number of iterations specified. This is usually the
    // case in very noisy images
    if (homography.empty()) {
      std::cout << "skipping frame, no homography found\n";
      continue;
    }

    cv::perspectiveTransform(source_corners, videoframe_corners, homography);

    // incredible fast but simple discarding of bad posing
    if (!cv::isContourConvex(videoframe_corners)) {
      std::cout << "Contour not convex, matching source impossible\n";
      continue;
    }

    // debug-drawing of keypoints
    // cv::drawKeypoints(videoframe, key_points_videoframe,
    // debug_img_videoframe);
    cv::drawMatches(source_image, keypoints_source, videoframe,
                    keypoints_videoframe, culled_matches, debug_img_videoframe,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // drawMatches puts the source_image on the left side of the output image
    // so we need to shift the matching box to the right by source_image.width
    for (auto &c : videoframe_corners) {
      c += cv::Point2f((float)source_image.cols, 0);
    }
    // draw matching box
    cv::line(debug_img_videoframe, videoframe_corners[0], videoframe_corners[1],
             cv::Scalar(255, 0, 0), 4);
    cv::line(debug_img_videoframe, videoframe_corners[1], videoframe_corners[2],
             cv::Scalar(0, 255, 0), 4);
    cv::line(debug_img_videoframe, videoframe_corners[2], videoframe_corners[3],
             cv::Scalar(255, 255, 0), 4);
    cv::line(debug_img_videoframe, videoframe_corners[3], videoframe_corners[0],
             cv::Scalar(0, 0, 255), 4);

    cv::imshow("Cam output", debug_img_videoframe); // put the image on screen

    // for a keypress and exit if any detected
    auto killer_key = cv::waitKey(1);
    if (killer_key >= 0 && killer_key < 255) {
      break;
    }
  }
}
