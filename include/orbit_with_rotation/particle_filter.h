#pragma once
#include <orbit_with_rotation/observation.h>
#include <orbit_with_rotation/particle_filter_configuration_parameters.h>
#include <orbit_with_rotation/prediction.h>

#include <memory>

namespace orbit_with_rotation {

class particle_filter {
 private:
  struct impl;

  struct impl_deleter {
    void operator()(impl* p_impl);
  };

  std::unique_ptr<impl, impl_deleter> p_impl_;

 public:
  [[nodiscard]] prediction extrapolate_state(const float& time_offset_seconds) const noexcept;

  void update_state_sans_observation(const float& time_offset_seconds) noexcept;
  void update_state_with_observation(const float& time_offset_seconds, const observation& observation_state) noexcept;

  particle_filter(
      const std::size_t& number_of_particles,
      const observation& initial_observation,
      const particle_filter_configuration_parameters& params) noexcept;
};

}  // namespace orbit_with_rotation
