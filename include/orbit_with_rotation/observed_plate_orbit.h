#pragma once

#include <Eigen/Dense>

namespace orbit_with_rotation {

struct observed_plate_orbit {
  float radius;
  float orientation;
  Eigen::Vector3f center;
};

}
