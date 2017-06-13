#ifndef FEATUREMATCHER_H
#define FEATUREMATCHER_H

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
// fwd decl
struct SURF_Image;

class FeatureMatcher {
public:
  /// compute possible match for occurence of 'source_image' in 'video_frame'
  /// after the execution the coords, field of the input frames will be filled
  /// returns true on success, false otherwise
  bool computeMatches(SURF_Image &source_image, SURF_Image &video_frame);

  /// returns the list of matches
  const std::vector<cv::DMatch> &matches() const { return _culled_matches; }

  /// the minimum of culled matches for the computation to be valid
  size_t matchingThreshold() const { return _matching_threshold; }
  void setMatchingThreshold(const size_t &matching_threshold) {
    _matching_threshold = matching_threshold;
  }

private:
  cv::FlannBasedMatcher _matcher;
  std::vector<cv::DMatch> _culled_matches;
  size_t _matching_threshold{5};
};

#endif // FEATUREMATCHER_H
