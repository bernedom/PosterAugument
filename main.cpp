#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "featurematcher.h"
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

bool doContinue() {
  // for a keypress and exit if any detected
  const auto killer_key = cv::waitKey(1);
  if (killer_key >= 0 && killer_key < 255) {
    return false;
  }
  return true;
}

void draw_plain(const SURF_Image &source_image, const SURF_Image &video_frame,
                const char *msg) {

  const auto height =
      std::max(source_image.raw_data.rows, video_frame.raw_data.rows);

  cv::Mat result(height, source_image.raw_data.cols + video_frame.raw_data.cols,
                 source_image.raw_data.type());
  auto left_roi = result(
      cv::Rect(0, 0, source_image.raw_data.cols, source_image.raw_data.rows));
  auto right_roi =
      result(cv::Rect(source_image.raw_data.cols, 0, video_frame.raw_data.cols,
                      video_frame.raw_data.rows));

  source_image.raw_data.copyTo(left_roi);
  video_frame.raw_data.copyTo(right_roi);

  cv::putText(result, msg, cv::Point(50, result.rows - 50),
              cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(0, 255, 255));

  cv::imshow("Cam output", result);
  std::cout << "FAILED: " << msg << std::endl;
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

  SURF_Image replacement_image;
  replacement_image.raw_data =
      cv::imread("C:\\Code\\PosterAugument\\Assets\\pirate_smiley.jpg");

  if (replacement_image.raw_data.cols > source_image.raw_data.cols ||
      replacement_image.raw_data.rows > source_image.raw_data.rows) {
    std::cerr << "Replacement image must fit inside source image\n";
    return -1;
  }

  const auto height_offset =
      (source_image.raw_data.rows - replacement_image.raw_data.rows) / 2;
  const auto width_offset =
      (source_image.raw_data.cols - replacement_image.raw_data.cols) / 2;
  replacement_image.corners = {
      cv::Point2f(width_offset, height_offset),
      cv::Point2f((float)replacement_image.raw_data.cols + width_offset,
                  height_offset),
      cv::Point2f((float)replacement_image.raw_data.cols + width_offset,
                  (float)replacement_image.raw_data.rows + height_offset),
      cv::Point2f(width_offset,
                  (float)replacement_image.raw_data.rows + height_offset)};

  const auto min_hessian = 400;
  cv::Ptr<cv::xfeatures2d::SURF> detector =
      cv::xfeatures2d::SURF::create(min_hessian);

  compute_surf(source_image, detector);

  SURF_Image video_frame;
  FeatureMatcher matcher;

  cv::Mat debug_img_videoframe;

  while (doContinue()) {

    stream.read(video_frame.raw_data); // slurp a single frame from the webcam

    compute_surf(video_frame, detector);

    // this happens if a featureless frame is recorded, i.e. from overexposure
    // or a blocked camera
    if (video_frame.descriptors.cols == 0 ||
        video_frame.descriptors.rows == 0) {
      draw_plain(source_image, video_frame,
                 "Not enough features to perform matching, skipping frame");
      continue;
    }

    if (!matcher.computeMatches(source_image, video_frame)) {
      draw_plain(source_image, video_frame, "Not enough matching features");
      continue;
    }

    // creating a set of homogenous transforms between points to do pose
    // estimation, limiting the number of RANSAC iterations so it performs a
    // bit faster
    cv::Mat homography =
        cv::findHomography(source_image.coords, video_frame.coords, CV_RANSAC,
                           3, cv::noArray(), 1000);

    // this might fail for no aparent reason because RANSAC might not find a
    // consensus given the number of iterations specified. This is usually the
    // case in very noisy images
    if (homography.empty()) {
      draw_plain(source_image, video_frame,
                 "skipping frame, no homography found");
      continue;
    }

    cv::perspectiveTransform(source_image.corners, video_frame.corners,
                             homography);

    // incredible fast but simple discarding of bad posing :)
    if (!cv::isContourConvex(video_frame.corners)) {
      draw_plain(source_image, video_frame,
                 "Contour not convex, matching source impossible");
      continue;
    }

    // debug-drawing of keypoints
    // cv::drawKeypoints(videoframe, key_points_videoframe,
    // debug_img_videoframe);
    cv::drawMatches(
        source_image.raw_data, source_image.keypoints, video_frame.raw_data,
        video_frame.keypoints, matcher.matches(), debug_img_videoframe,
        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
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

    cv::Mat distorted_image;
    cv::warpPerspective(replacement_image.raw_data, distorted_image, homography,
                        cv::Size(video_frame.cols, video_frame.rows));
    cv::imshow("Distorted", distorted_image);

    //    auto roi =
    //        debug_img_videoframe(cv::Rect(0, 0,
    //        replacement_image.raw_data.cols,
    //                                      replacement_image.raw_data.rows));
    //    distorted_image.copyTo(roi);

    cv::imshow("Cam output", debug_img_videoframe); // put the image on screen
  }
}
