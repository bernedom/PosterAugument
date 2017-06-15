#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <brofiler.h>

#include "augumentor_cpu.h"
#include "featurematcher.h"
#include "surf_image.h"

#include <algorithm>
#include <iostream>
#include <vector>

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

  BROFILER_FRAME("INIT")
  BROFILER_CATEGORY("Init Frame", Profiler::Color::AliceBlue);
  BROFILER_CATEGORY("Opening stream", Profiler::Color::Aqua);
  cv::VideoCapture stream(0); // no const, because read() is not const
  if (!stream.isOpened()) {
    std::cerr << "Could not open camera\n";
    return -1;
  }
  BROFILER_CATEGORY("reading images stream", Profiler::Color::Aqua);
  SURF_Image source_image;
  source_image.raw_data =
      cv::imread("C:\\Code\\PosterAugument\\Assets\\JollyRoger.jpg");

  SURF_Image replacement_image;
  replacement_image.raw_data = cv::imread(
      "C:\\Code\\PosterAugument\\Assets\\pirate_smiley_transparent.png",
      cv::IMREAD_UNCHANGED); // reading this with transparency information

  Augumentor_CPU augumentor(std::move(source_image),
                            std::move(replacement_image));

  BROFILER_CATEGORY("Augumentor Init", Profiler::Color::Aqua);
  if (!augumentor.init()) {
    return -1;
  }

  SURF_Image video_frame;

  cv::Mat debug_img_videoframe;

  while (input_handler(augumentor)) {

    BROFILER_FRAME("MainLoop");

    BROFILER_EVENT("Reading video frame")
    stream.read(video_frame.raw_data); // slurp a single frame from the webcam

    BROFILER_EVENT("Compute Analysis")
    augumentor.compute(video_frame);

    BROFILER_EVENT("Render")
    augumentor.render(video_frame);

    BROFILER_EVENT("Outputting Image")
    cv::imshow("Cam output", video_frame.raw_data);
  }
}
