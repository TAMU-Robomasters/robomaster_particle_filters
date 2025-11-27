#pragma once

#include <orbit_with_rotation/observed_plate.h>
#include <orbit_with_rotation/observed_plate_orbit.h>
#include <thrust/execution_policy.h>

#include <Eigen/Dense>
#include <cmath>

namespace orbit_with_rotation {

class observed_plate_orbit_builder {
 private:
  float radius_prior_;
  Eigen::Vector3f observer_position_;

 public:
  PF_TARGET_ATTRS observed_plate_orbit from_one_plate(const observed_plate& plate_one) const noexcept {
    const float rotation = plate_one.rotation()[0];
    const float sin_rot = std::sinf(rotation);
    const float cos_rot = std::cosf(rotation);

    const Eigen::Vector3f radius_vector{radius_prior_ * sin_rot, radius_prior_ * cos_rot, 0.0f};
    const Eigen::Vector3f predicted_center = plate_one.position() + radius_vector;

    return observed_plate_orbit{radius_prior_, rotation, predicted_center};
  }

  PF_TARGET_ATTRS observed_plate_orbit
  from_two_plates(const observed_plate& plate_one, const observed_plate& plate_two) const noexcept {
    const Eigen::Vector3f midpoint = 0.5f * (plate_one.position() + plate_two.position());
    const Eigen::Vector3f one_to_two = plate_two.position() - plate_one.position();
    const Eigen::Vector3f delta = 0.5f * (Eigen::Vector3f{} << -one_to_two[1], one_to_two[0], 0.0f).finished();

    const float rotation = plate_one.rotation()[0];
    const float sin_rot = std::sinf(rotation);
    const float cos_rot = std::cosf(rotation);
    const Eigen::Vector3f plate_normal{sin_rot, cos_rot, 0.0f};
    const float signed_side = plate_normal.dot(delta);

    const Eigen::Vector3f candidate_pos = midpoint + delta;
    const Eigen::Vector3f mirrored_pos = midpoint - delta;

    const Eigen::Vector3f predicted_center =
        (std::abs(signed_side) > 1e-5f)
            ? ((signed_side >= 0.0f) ? candidate_pos : mirrored_pos)
            : (((candidate_pos - observer_position_).squaredNorm() < (mirrored_pos - observer_position_).squaredNorm())
                   ? candidate_pos
                   : mirrored_pos);

    const float predicted_radius = M_SQRT2 * delta.norm();

    return observed_plate_orbit{predicted_radius, rotation, predicted_center};
  }

  PF_TARGET_ATTRS observed_plate_orbit_builder(const float& radius_prior, const Eigen::Vector3f& observer_position) noexcept
      : radius_prior_{radius_prior}, observer_position_{observer_position} {}
};

}  // namespace plate_orbit
