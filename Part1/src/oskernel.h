#ifndef OSKERNEL_H
#define OSKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

// Steer with this first
__device__ 
glm::vec3 steerForSeek(glm::vec4 my_pos, glm::vec3 my_vel, glm::vec4 my_target);

// Highest magnitude vector is closest. Use that one for steering 
__device__
glm::vec3 steerToAvoidNeighbor(glm::vec3 my_pos, glm::vec3 my_vel, glm::vec3 their_pos, glm::vec3 their_vel);

// Call first for all, keep track of # non-zero vals
// If non-zero, normalize, else, call steer to avoid neighbor
__device__
glm::vec3 steerToAvoidCloseNeighbor(glm::vec3 my_pos, glm::vec3 my_vel, glm::vec3 their_pos, glm::vec3 their_vel, float maxDist, float cosMaxAngle);

__device__
float predictNearestApproachTime(glm::vec3 my_pos, glm::vec3 my_vel, glm::vec3 their_pos, glm::vec3 their_vel);

__device__
float computeNearestApproachPositions(glm::vec3 my_pos, glm::vec3 my_vel, glm::vec3 their_pos, glm::vec3 their_vel, float time);

__device__
glm::vec3 adjustRowSteeringForce(glm::vec3 my_pos, glm::vec3 my_vel, glm::vec3 force);

#endif