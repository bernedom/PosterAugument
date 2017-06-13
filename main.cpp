#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "surf_image.h"

#include <algorithm>
#include <iostream>
#include <vector>

void compute_surf(SURF_Image &image,
                  const cv::Ptr<cv::xfeatures2d::SURF> &detector) {
  image.keypoints.clear();
  image.coords.clear();
  detector->detect(image.raw_data, image.keypoints);
  detector->compute(image.raw_data, image.keypoints, image.descriptors);
}

int main(int, char **) {

  cv::VideoCapture stream(0); // no const, because read() is not const
  if (!stream.isOpened()) {
    std::cerr << "Could not open camera\n";
    return -1;
  }

  SURF_Image source_image;
  source_image.raw_data =
      cv::imread("C:\\Code\\PosterAugument\\Assets\\JollyRoger.jpg");

  source_image.corners = {cv::Point2f(0, 0),
                          cv::Point2f((float)source_image.raw_data.cols, 0),
                          cv::Point2f((float)source_image.raw_data.cols,
                                      (float)source_image.raw_data.rows),
                          cv::Point2f(0, (float)source_image.raw_data.rows)};

  const auto min_hessian = 400;
  cv::Ptr<cv::xfeatures2d::SURF> detector =
      cv::xfeatures2d::SURF::create(min_hessian);

  compute_surf(source_image, detector);

  SURF_Image video_frame;

  cv::Mat debug_img_videoframe;

  while (true) {

    stream.read(video_frame.raw_data); // slurp a single frame from the webcam

    compute_surf(video_frame, detector);

    if (video_frame.descriptors.cols == 0 ||
        video_frame.descriptors.rows == 0) {
      std::cout << "Not enough features to perform matching, skipping frame\n";
      continue;
    }

    // find matches between the keypoints
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(source_image.descriptors, video_frame.descriptors, matches);

    // get the min & max distance (max is not
    const auto cmp_func = [](const cv::DMatch &lhs, const cv::DMatch &rhs) {
      return lhs.distance < rhs.distance;
    };
    // this will frequently yield 0 which is a good indicator for matching
    // points
    const auto min_distance =
        std::min_element(matches.begin(), matches.end(), cmp_func);

    source_image.coords.clear();
    video_frame.coords.clear();

    // this culling using a multiplier is to better find fuzzy matches (having
    // min_dist > 0)
    const double max_distance_multiplier = 3.0;
    std::vector<cv::DMatch> culled_matches;
    for (const auto &m : matches) {
      if (m.distance < min_distance->distance * max_distance_multiplier) {
        culled_matches.push_back(m);
        source_image.coords.push_back(source_image.keypoints[m.queryIdx].pt);
        video_frame.coords.push_back(video_frame.keypoints[m.trainIdx].pt);
      }
    }

    std::cout << culled_matches.size() << std::endl;

    // creating a set of homogenous transforms between points to do pose
    // estimation, limiting the number of RANSAC iterations so it performs a bit
    // faster
    cv::Mat homography =
        cv::findHomography(source_image.coords, video_frame.coords, CV_RANSAC,
                           3, cv::noArray(), 1000);

    // this might fail for no aparent reason because RANSAC might not find a
    // consensus given the number of iterations specified. This is usually the
    // case in very noisy images
    if (homography.empty()) {
      std::cout << "skipping frame, no homography found\n";
      continue;
    }

    cv::perspectiveTransform(source_image.corners, video_frame.corners,
                             homography);

    // incredible fast but simple discarding of bad posing
    if (!cv::isContourConvex(video_frame.corners)) {
      std::cout << "Contour not convex, matching source impossible\n";
      continue;
    }

    // debug-drawing of keypoints
    // cv::drawKeypoints(videoframe, key_points_videoframe,
    // debug_img_videoframe);
    cv::drawMatches(source_image.raw_data, source_image.keypoints,
                    video_frame.raw_data, video_frame.keypoints, culled_matches,
                    debug_img_videoframe, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // drawMatches puts the source_image on the left side of the output image
    // so we need to shift the matching box to the right by source_image.width
    for (auto &c : video_frame.corners) {
      c += cv::Point2f((float)source_image.raw_data.cols, 0);
    }
    // draw matching box
    cv::line(debug_img_videoframe, video_frame.corners[0],
             video_frame.corners[1], cv::Scalar(255, 0, 0), 4);
    cv::line(debug_img_videoframe, video_frame.corners[1],
             video_frame.corners[2], cv::Scalar(0, 255, 0), 4);
    cv::line(debug_img_videoframe, video_frame.corners[2],
             video_frame.corners[3], cv::Scalar(255, 255, 0), 4);
    cv::line(debug_img_videoframe, video_frame.corners[3],
             video_frame.corners[0], cv::Scalar(0, 0, 255), 4);

    cv::imshow("Cam output", debug_img_videoframe); // put the image on screen

    // for a keypress and exit if any detected
    auto killer_key = cv::waitKey(1);
    if (killer_key >= 0 && killer_key < 255) {
      break;
    }
  }
}
