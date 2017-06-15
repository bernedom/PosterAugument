#include "featurematcher.h"
#include "surf_image.h"

#include <Brofiler.h>

bool FeatureMatcher::computeMatches(SURF_Image &source_image,
                                    SURF_Image &video_frame) {

  BROFILER_CATEGORY("FeatureMatcher::computingMatches",
                    Profiler::Color::DarkBlue)
  std::vector<cv::DMatch> matches;
  // find matches between the keypoints
  BROFILER_EVENT("Mattching descriptors")
  _matcher.match(source_image.descriptors, video_frame.descriptors, matches);

  // get the min & max distance (max is not
  const auto cmp_func = [](const cv::DMatch &lhs, const cv::DMatch &rhs) {
    return lhs.distance < rhs.distance;
  };
  // this will frequently yield 0 which is a good indicator for matching
  // points
  const auto min_distance =
      std::min_element(matches.begin(), matches.end(), cmp_func);

  source_image.coords.clear();
  video_frame.coords.clear();
  _culled_matches.clear();

  // this culling using a multiplier is to better find fuzzy matches (having
  // min_dist > 0)
  const double max_distance_multiplier = 3.0;
  BROFILER_EVENT("Culling matches");
  for (const auto &m : matches) {
    if (m.distance < min_distance->distance * max_distance_multiplier) {
      _culled_matches.push_back(m);
      source_image.coords.push_back(source_image.keypoints[m.queryIdx].pt);
      video_frame.coords.push_back(video_frame.keypoints[m.trainIdx].pt);
    }
  }

  const size_t matching_threshold = 5;
  if (_culled_matches.size() < matching_threshold) {
    return false;
  }
  return true;
}
