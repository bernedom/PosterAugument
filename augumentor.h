#ifndef AUGUMENTOR_H
#define AUGUMENTOR_H

#include "surf_image.h"

class Augumentor {
public:
  Augumentor() = delete;
  Augumentor(SURF_Image &&source_image, SURF_Image &&replacement_image)
      : _source_image(source_image), _replacement_image(replacement_image) {}

  virtual bool init() = 0;
  virtual void compute(SURF_Image &video_frame) = 0;
  virtual void render(SURF_Image &video_frame) = 0;

  void toggle_renderdebug() { _render_debug = !_render_debug; }
  void toggle_animation() { _animate = !_animate; }

  bool renderdebug() const { return _render_debug; }
  bool animate() const { return _animate; }

protected:
  SURF_Image _source_image;
  SURF_Image _replacement_image;

private:
  bool _render_debug{false};
  bool _animate{false};
};

#endif // AUGUMENTOR_H
