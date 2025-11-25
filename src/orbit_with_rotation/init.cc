// plate orbit
#include <orbit_with_rotation/observation.h>
#include <orbit_with_rotation/observed_plate.h>
#include <orbit_with_rotation/particle_filter.h>
#include <orbit_with_rotation/particle_filter_configuration_parameters.h>
#include <orbit_with_rotation/predicted_plate.h>
#include <orbit_with_rotation/prediction.h>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace orbit_with_rotation {

namespace py = pybind11;

void init(py::module_& m) noexcept {
  auto orbit_with_rotation = m.def_submodule("orbit_with_rotation");

  py::class_<observed_plate>(orbit_with_rotation, "ObservedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &observed_plate::position)
      .def("position_diagonal_covariance", &observed_plate::position_diagonal_covariance);

  py::class_<predicted_plate>(orbit_with_rotation, "PredictedPlate")
      .def(py::init<Eigen::Vector3f, Eigen::Vector3f>())
      .def("position", &predicted_plate::position)
      .def("velocity", &predicted_plate::velocity);

  py::class_<observation>(orbit_with_rotation, "Observation")
      .def_static("from_one_plate", &observation::from_one_plate)
      .def_static("from_two_plates", &observation::from_two_plates);

  py::class_<prediction>(orbit_with_rotation, "Prediction")
      .def("predicted_plates", &prediction::predicted_plates_for_host)
      .def("radius", &prediction::radius)
      .def("orientation", &prediction::orientation)
      .def("orientation_velocity", &prediction::orientation_velocity)
      .def("center", &prediction::center)
      .def("center_velocity", &prediction::center_velocity)
      .def("extrapolate_state", &prediction::extrapolate_state);

  py::class_<particle_filter_configuration_parameters>(orbit_with_rotation, "ParticleFilterConfigurationParameters")
      .def(py::init<float, float, float, float, float, float, float, float, Eigen::Vector2f, Eigen::Vector2f>());

  py::class_<particle_filter>(orbit_with_rotation, "ParticleFilter")
      .def(py::init<size_t, observation, particle_filter_configuration_parameters>())
      .def("extrapolate_state", &particle_filter::extrapolate_state, py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_sans_observation",
          &particle_filter::update_state_sans_observation,
          py::call_guard<py::gil_scoped_release>())

      .def(
          "update_state_with_observation",
          &particle_filter::update_state_with_observation,
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace orbit_with_rotation
