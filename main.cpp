#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "augumentor_cpu.h"
#include "featurematcher.h"
#include "surf_image.h"

#include <algorithm>
#include <iostream>
#include <vector>

static bool draw_debug = false;
static bool rotate_replacement = false;

bool input_handler(Augumentor &augumentor) {
  // for a keypress and exit if any detected
  const auto killer_key = cv::waitKey(1);
  bool result = true;
  switch (killer_key) {
  case 100: {
    augumentor.toggle_renderdebug();
    break;
  }
  case 114:
    augumentor.toggle_animation();
    break;
  case 255:
  case -1:
    result = true;
    break;
  default:
    result = false;
  }

  return result;
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

  SURF_Image replacement_image;
  replacement_image.raw_data = cv::imread(
      "C:\\Code\\PosterAugument\\Assets\\pirate_smiley_transparent.png",
      cv::IMREAD_UNCHANGED); // reading this with transparency information

  Augumentor_CPU augumentor(std::move(source_image),
                            std::move(replacement_image));

  if (!augumentor.init()) {
    return -1;
  }

  SURF_Image video_frame;

  cv::Mat debug_img_videoframe;

  while (input_handler(augumentor)) {

    stream.read(video_frame.raw_data); // slurp a single frame from the webcam

    augumentor.compute(video_frame);

    augumentor.render(video_frame);

    cv::imshow("Cam output", video_frame.raw_data);
  }
}
