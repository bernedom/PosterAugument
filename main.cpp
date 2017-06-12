#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

int main(int, char **) {

  cv::VideoCapture stream(0); // no const, because read() is not const

  const auto source_image =
      cv::imread("C:\\Code\\PosterAugument\\Assets\\JollyRoger.jpg");

  if (!stream.isOpened()) {
    std::cerr << "Could not open camera\n";
    return -1;
  }

  while (true) {
    cv::Mat frame;
    stream.read(frame); // slurp a single frame from the webcam

    cv::imshow("Cam output", frame); // put the image on screen

    // wait for 30ms for a keypress and exit if any detected
    auto killer_key = cv::waitKey(30);
    if (killer_key >= 0 && killer_key < 255) {
      std::cout << "Killer key " << killer_key << std::endl;
      break;
    }
  }
}
