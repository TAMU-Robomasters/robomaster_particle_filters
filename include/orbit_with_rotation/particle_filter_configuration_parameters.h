#pragma once

#include <Eigen/Dense>

namespace orbit_with_rotation {

struct particle_filter_configuration_parameters {
  float radius_prior;
  float visibility_logit_coefficient;
  
  float radius_prior_variance_one_plate;
  float radius_prior_variance_two_plates;
  float radius_process_variance;
  
  float orientation_velocity_prior_variance;
  float orientation_velocity_process_variance;
  
  float center_z_position_process_variance;
  Eigen::Vector2f center_xy_velocity_prior_diagonal_covariance;
  Eigen::Vector2f center_xy_velocity_process_diagonal_covariance;
};

}  // namespace orbit_with_rotation
