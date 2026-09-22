cmake_minimum_required(VERSION 3.8)
project(fr5_ag160_95_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
