#include <QCoreApplication>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

int main(int, char **) {

  cv::VideoCapture stream(0);

  if (!stream.isOpened()) {
    std::cerr << "Could not open camera\n";
    return -1;
  }

  while (true) {
    cv::Mat frame;
    stream.read(frame);

    cv::imshow("Cam output", frame);
    auto killer_key = cv::waitKey(30);
    if (killer_key >= 0 && killer_key < 255) {
      std::cout << "Killer key " << killer_key << std::endl;
      break;
    }
  }
}
