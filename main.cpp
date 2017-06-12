#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp >

#include <iostream>
#include <vector>

int main(int, char **) {

  cv::VideoCapture stream(0); // no const, because read() is not const

  const auto source_image =
      cv::imread("C:\\Code\\PosterAugument\\Assets\\JollyRoger.jpg");

  const auto min_hessian = 400;
  cv::Ptr<cv::xfeatures2d::SURF> detector =
      cv::xfeatures2d::SURF::create(min_hessian);

  std::vector<cv::KeyPoint> key_points_source;
  std::vector<cv::KeyPoint> key_points_input;

  detector->detect(source_image, key_points_source);

  if (!stream.isOpened()) {
    std::cerr << "Could not open camera\n";
    return -1;
  }

  while (true) {
    key_points_input.clear();

    cv::Mat frame;
    stream.read(frame); // slurp a single frame from the webcam
    detector->detect(frame, key_points_input);

    cv::imshow("Cam output", frame); // put the image on screen

    // wait for 30ms for a keypress and exit if any detected
    auto killer_key = cv::waitKey(30);
    if (killer_key >= 0 && killer_key < 255) {
      std::cout << "Killer key " << killer_key << std::endl;
      break;
    }
  }
}
