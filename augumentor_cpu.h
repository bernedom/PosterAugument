#ifndef AUGUMENTOR_CPU_H
#define AUGUMENTOR_CPU_H

#include "augumentor.h"
#include "featurematcher.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <string>

// CPU based implementation for augmentation
class Augumentor_CPU : public Augumentor {
public:
  Augumentor_CPU() = delete;
  Augumentor_CPU(SURF_Image &&source_image, SURF_Image &&replacement_image)
      : Augumentor(std::move(source_image), std::move(replacement_image)){};

  virtual bool init() override;

  virtual void compute(SURF_Image &video_frame) override;
  virtual void render(SURF_Image &video_frame) override;

private:
  void compute_surf(SURF_Image &image);
  void warp_replacement_image(SURF_Image &video_frame,
                              const cv::Mat &homography);

private:
  cv::Ptr<cv::xfeatures2d::SURF> _detector;
  FeatureMatcher _matcher;
  std::string _error_msg;
};

#endif // AUGUMENTOR_CPU_H
