#pragma once

#include <orbit_with_rotation/observation.h>
#include <orbit_with_rotation/predicted_plate.h>
#include <pf/util/device_array.h>

#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace orbit_with_rotation {

namespace helper {

PF_TARGET_ATTRS [[nodiscard]] inline float to_radius(const float& radius) noexcept {
  constexpr float min_radius = 0.05f;
  return thrust::max(min_radius, radius);
}

PF_TARGET_ATTRS [[nodiscard]] inline float to_orientation(const float& angle_radians) noexcept {
  const float value = fmod(angle_radians, M_PI_2);
  return (value < 0.0f) ? value + M_PI_2 : value;
}

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f rpad_zero(const Eigen::Vector2f& vector) noexcept {
  return (Eigen::Vector3f{} << vector, Eigen::Matrix<float, 1, 1>::Zero()).finished();
}

PF_TARGET_ATTRS [[nodiscard]] inline Eigen::Vector3f lpad_zero(const float& scalar) noexcept {
  return (Eigen::Vector3f{} << 0.0f, 0.0f, scalar).finished();
}

}  // namespace helper

class prediction {
 private:
  static constexpr size_t number_of_plates = 4;

  float radius_;

  float orientation_;
  float orientation_velocity_;

  Eigen::Vector3f center_;
  Eigen::Vector2f center_velocity_;

  struct orientation_offset {
    float cos_coeff;
    float sin_coeff;
    float radius;
  };

 public:
  [[nodiscard]] std::array<predicted_plate, number_of_plates> predicted_plates_for_host() const noexcept {
    return predicted_plates().to_host_array();
  }

  PF_TARGET_ATTRS [[nodiscard]] pf::util::device_array<predicted_plate, number_of_plates> predicted_plates() const noexcept {
    const float sin_orientation = sinf(orientation_);
    const float cos_orientation = cosf(orientation_);

    const pf::util::device_array<orientation_offset, number_of_plates> orientation_offsets = {
        orientation_offset{1.0f, 0.0f, radius_},
        orientation_offset{0.0f, 1.0f, radius_},
        orientation_offset{-1.0f, 0.0f, radius_},
        orientation_offset{0.0f, -1.0f, radius_},
    };

    return orientation_offsets.transformed([this, sin_orientation, cos_orientation](const orientation_offset& offset) {
      const float cos_angle = cos_orientation * offset.cos_coeff - sin_orientation * offset.sin_coeff;
      const float sin_angle = sin_orientation * offset.cos_coeff + cos_orientation * offset.sin_coeff;

      const Eigen::Vector3f predicted_plate_position =
          center_ + offset.radius * (Eigen::Vector3f{} << cos_angle, sin_angle, 0.0f).finished();

      const Eigen::Vector3f predicted_plate_velocity =
          helper::rpad_zero(center_velocity_) +
          orientation_velocity_ * offset.radius * (Eigen::Vector3f{} << -sin_angle, cos_angle, 0.0f).finished();

      return predicted_plate(predicted_plate_position, predicted_plate_velocity);
    });
  }

  PF_TARGET_ATTRS [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept {
    return prediction(
        radius_,
        orientation_ + time_offset_seconds * orientation_velocity_,
        orientation_velocity_,
        center_ + time_offset_seconds * helper::rpad_zero(center_velocity_),
        center_velocity_);
  }

  PF_TARGET_ATTRS void update_state(
      const float& time_offset_seconds,
      const float& radius_noise,
      const float& orientation_velocity_noise_0,
      const float& orientation_velocity_noise_1,
      const float& center_z_position_noise,
      const Eigen::Vector2f& center_xy_velocity_noise_0,
      const Eigen::Vector2f& center_xy_velocity_noise_1) noexcept {
    static constexpr float one_half = 1.0 / 2.0;
    static constexpr float one_twelfth = 1.0 / 12.0;

    const float radius_noise_scale = sqrtf(time_offset_seconds);
    const float velocity_noise_scale = radius_noise_scale;
    const float center_z_position_noise_scale = radius_noise_scale;
    const float position_noise_scale = sqrtf(one_twelfth) * powf(velocity_noise_scale, 3);

    const float d_radius = radius_noise_scale * radius_noise;
    const float d_center_z = center_z_position_noise_scale * center_z_position_noise;

    const float d_orientation_velocity = velocity_noise_scale * orientation_velocity_noise_1;
    const float d_orientation = time_offset_seconds * orientation_velocity_ +
                                one_half * time_offset_seconds * d_orientation_velocity +
                                position_noise_scale * orientation_velocity_noise_0;

    const Eigen::Vector2f d_center_velocity = velocity_noise_scale * center_xy_velocity_noise_1;
    const Eigen::Vector3f d_center = time_offset_seconds * helper::rpad_zero(center_velocity_) +
                                     one_half * time_offset_seconds * helper::rpad_zero(d_center_velocity) +
                                     position_noise_scale * helper::rpad_zero(center_xy_velocity_noise_0) +
                                     helper::lpad_zero(d_center_z);

    radius_ = helper::to_radius(radius_ + d_radius);

    orientation_ = helper::to_orientation(orientation_ + d_orientation);
    orientation_velocity_ = orientation_velocity_ + d_orientation_velocity;

    center_ = center_ + d_center;
    center_velocity_ = center_velocity_ + d_center_velocity;
  }

  PF_TARGET_ATTRS [[nodiscard]] const float& radius() const noexcept { return radius_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation() const noexcept { return orientation_; }
  PF_TARGET_ATTRS [[nodiscard]] const float& orientation_velocity() const noexcept { return orientation_velocity_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector3f& center() const noexcept { return center_; }
  PF_TARGET_ATTRS [[nodiscard]] const Eigen::Vector2f& center_velocity() const noexcept { return center_velocity_; }

  PF_TARGET_ATTRS prediction() noexcept
      : radius_{0.0f},
        orientation_{0.0f},
        orientation_velocity_{0.0f},
        center_{Eigen::Vector3f::Zero()},
        center_velocity_{Eigen::Vector2f::Zero()} {}

  PF_TARGET_ATTRS prediction(
      const float& radius,
      const float& orientation,
      const float& orientation_velocity,
      const Eigen::Vector3f& center,
      const Eigen::Vector2f& center_velocity) noexcept
      : radius_{helper::to_radius(radius)},
        orientation_{helper::to_orientation(orientation)},
        orientation_velocity_{orientation_velocity},
        center_{center},
        center_velocity_{center_velocity} {}
};

}  // namespace orbit_with_rotation
