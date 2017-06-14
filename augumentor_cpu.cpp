#include "augumentor_cpu.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <iostream>

bool Augumentor_CPU::init() {
  _source_image.corners = {cv::Point2f(0, 0),
                           cv::Point2f((float)_source_image.raw_data.cols, 0),
                           cv::Point2f((float)_source_image.raw_data.cols,
                                       (float)_source_image.raw_data.rows),
                           cv::Point2f(0, (float)_source_image.raw_data.rows)};

  if (_replacement_image.raw_data.cols > _source_image.raw_data.cols ||
      _replacement_image.raw_data.rows > _source_image.raw_data.rows) {
    std::cerr << "Replacement image must fit inside source image\n";
    return false;
  }

  const auto min_hessian = 400;
  _detector = cv::xfeatures2d::SURF::create(min_hessian);

  compute_surf(_source_image);
  return true;
}

void Augumentor_CPU::compute(SURF_Image &video_frame) {
  _error_msg.clear();
  compute_surf(video_frame);
  // this happens if a featureless frame is recorded, i.e. from overexposure
  // or a blocked camera
  if (video_frame.descriptors.cols == 0 || video_frame.descriptors.rows == 0) {
    _error_msg = "Not enough features to perform matching, skipping frame";
    return;
  }
  if (!_matcher.computeMatches(_source_image, video_frame)) {
    _error_msg = "Not enough matching features";
    return;
  }

  // creating a set of homogenous transforms between points to do pose
  // estimation, limiting the number of RANSAC iterations so it performs a
  // bit faster
  cv::Mat homography =
      cv::findHomography(_source_image.coords, video_frame.coords, CV_RANSAC, 3,
                         cv::noArray(), 500);

  // this might fail for no aparent reason because RANSAC might not find a
  // consensus given the number of iterations specified. This is usually the
  // case in very noisy images
  if (homography.empty()) {
    _error_msg = "skipping frame, no homography found";
    return;
  }

  cv::perspectiveTransform(_source_image.corners, video_frame.corners,
                           homography);

  // incredible fast but simple discarding of bad posing :)
  if (!cv::isContourConvex(video_frame.corners)) {
    _error_msg = "Contour not convex, matching source impossible";
    return;
  }

  warp_replacement_image(video_frame, homography);
}

void Augumentor_CPU::compute_surf(SURF_Image &image) {

  image.keypoints.clear();
  image.coords.clear();
  _detector->detect(image.raw_data, image.keypoints);
  _detector->compute(image.raw_data, image.keypoints, image.descriptors);
}

void Augumentor_CPU::render(SURF_Image &video_frame) {

  cv::Mat result;
  if (renderdebug()) {
    // emulate matching rendeirng if error ms is not empty
    if (!_error_msg.empty()) {

      const auto height =
          std::max(_source_image.raw_data.rows, video_frame.raw_data.rows);

      result = cv::Mat(height,
                       _source_image.raw_data.cols + video_frame.raw_data.cols,
                       _source_image.raw_data.type());
      auto left_roi = result(cv::Rect(0, 0, _source_image.raw_data.cols,
                                      _source_image.raw_data.rows));
      auto right_roi = result(cv::Rect(_source_image.raw_data.cols, 0,
                                       video_frame.raw_data.cols,
                                       video_frame.raw_data.rows));

      _source_image.raw_data.copyTo(left_roi);
      video_frame.raw_data.copyTo(right_roi);

      result.copyTo(video_frame.raw_data);
    } else {

      cv::Mat debug_image;
      cv::drawMatches(_source_image.raw_data, _source_image.keypoints,
                      video_frame.raw_data, video_frame.keypoints,
                      _matcher.matches(), debug_image, cv::Scalar::all(-1),
                      cv::Scalar::all(-1), std::vector<char>(),
                      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      video_frame.raw_data = debug_image;
      // drawMatches puts the source_image on the left side of the output image
      // so we need to shift the matching box to the right by source_image.width
      for (auto &c : video_frame.corners) {
        c += cv::Point2f((float)_source_image.raw_data.cols, 0);
      }
      // draw matching box
      cv::line(video_frame.raw_data, video_frame.corners[0],
               video_frame.corners[1], cv::Scalar(255, 0, 0), 4);
      cv::line(video_frame.raw_data, video_frame.corners[1],
               video_frame.corners[2], cv::Scalar(0, 255, 0), 4);
      cv::line(video_frame.raw_data, video_frame.corners[2],
               video_frame.corners[3], cv::Scalar(255, 255, 0), 4);
      cv::line(video_frame.raw_data, video_frame.corners[3],
               video_frame.corners[0], cv::Scalar(0, 0, 255), 4);
    }
  }

  if (!_error_msg.empty()) {
    cv::putText(video_frame.raw_data, _error_msg,
                cv::Point(50, video_frame.raw_data.rows - 50),
                cv::HersheyFonts::FONT_HERSHEY_PLAIN, 1.0,
                cv::Scalar(0, 255, 255));
    std::cout << "FAILED: " << _error_msg << std::endl;
  }
}

void Augumentor_CPU::warp_replacement_image(SURF_Image &video_frame,
                                            const cv::Mat &homography) {
  // distort replacement image according to homography

  cv::Mat distorted_image;

  if (animate()) {
    cv::Point2f center((float)_replacement_image.raw_data.cols / 2,
                       (float)_replacement_image.raw_data.rows / 2);

    static int angle = 0;
    const auto rot_mat = cv::getRotationMatrix2D(center, angle % 360, 0.6);
    angle += 10;
    cv::warpAffine(_replacement_image.raw_data, distorted_image, rot_mat,
                   cv::Size(_replacement_image.raw_data.cols,
                            _replacement_image.raw_data.rows));
    cv::warpPerspective(
        distorted_image, distorted_image, homography,
        cv::Size(video_frame.raw_data.cols, video_frame.raw_data.rows));
  } else {

    cv::warpPerspective(
        _replacement_image.raw_data, distorted_image, homography,
        cv::Size(video_frame.raw_data.cols, video_frame.raw_data.rows));
  }
  // assert to harden against implementation changes, as the loop below
  // relies on this
  assert(video_frame.raw_data.size == distorted_image.size);

  for (int c = 0; c < video_frame.raw_data.cols; ++c) {
    for (int r = 0; r < video_frame.raw_data.rows; ++r) {

      auto &video_px = video_frame.raw_data.at<cv::Vec3b>(r, c);
      const auto &overlay_px = distorted_image.at<cv::Vec4b>(r, c);

      const float alpha = overlay_px[3] / 255.0f;
      // blend using alpha weight
      for (int i = 0; i < 3; ++i) {
        video_px[i] =
            (uchar)(video_px[i] * (1.0f - alpha) + overlay_px[i] * alpha);
      }
    }
  }
}
