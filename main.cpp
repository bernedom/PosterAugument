#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
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

  const auto min_hessian = 400;
  cv::Ptr<cv::xfeatures2d::SURF> detector =
      cv::xfeatures2d::SURF::create(min_hessian);

  std::vector<cv::KeyPoint> key_points_source;
  cv::Mat descriptors_source;

  detector->detect(source_image, key_points_source);
  detector->compute(source_image, key_points_source, descriptors_source);

  std::cout << "Detected " << key_points_source.size()
            << " key points for source\n";

  cv::Mat debug_img_source;
  cv::drawKeypoints(source_image, key_points_source, debug_img_source);
  cv::imshow("source", debug_img_source); // debug drawing of keypoints

  std::vector<cv::KeyPoint> key_points_videoframe;
  cv::Mat debug_img_videoframe;
  stream.read(debug_img_videoframe); // read one frame to reserve image size

  while (true) {
    key_points_videoframe.clear();

    cv::Mat videoframe;
    cv::Mat descriptors_videoframe;

    stream.read(videoframe); // slurp a single frame from the webcam
    detector->detect(videoframe, key_points_videoframe);
    detector->compute(videoframe, key_points_videoframe,
                      descriptors_videoframe);

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

    // this culling using a multiplier is to better find fuzzy matches (having
    // min_dist < 0)
    const double max_distance_multiplier = 3.0;
    std::vector<cv::DMatch> culled_matches;
    for (const auto &m : matches) {
      if (m.distance < min_distance->distance * max_distance_multiplier) {
        culled_matches.push_back(m);
      }
    }

    // debug-drawing of keypoints
    cv::drawKeypoints(videoframe, key_points_videoframe, debug_img_videoframe);
    cv::imshow("Cam output", debug_img_videoframe); // put the image on screen

    // wait for 30ms for a keypress and exit if any detected
    auto killer_key = cv::waitKey(30);
    if (killer_key >= 0 && killer_key < 255) {
      break;
    }
  }
}
