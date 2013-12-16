#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "oskernel.h"
#include "utilities.h"
#include "const.h"
#include "device_functions.h"

using namespace glm;

// OpenSteer Device Code
__device__
vec3 steerForSeek(glm::vec4 my_pos, glm::vec3 my_vel, glm::vec4 my_target)
{
	vec4 t = (my_target - my_pos);
	vec3 vel = vec3(t.x,t.y,t.z) - my_vel;
	return vel;
}

__device__
// largest magnitude is closest. Only use that one for avoidance
vec3 steerToAvoidNeighbor(vec3 my_pos, vec3 my_vel, vec3 their_pos, vec3 their_vel, float minTime)
{
	const float collisionDangerThreshold = AVOID_RADIUS * 2;
	const float time = predictNearestApproachTime(my_pos, my_vel, their_pos, their_vel);
	float steer = 0;
	if (time >= 0 && time < minTime)
	{
		if (computeNearestApproachPositions(my_pos, my_vel, their_pos, their_vel)
		{
			float parallelness = dot(normalize(my_vel), normalize(their_vel));
			if (parallelness < -0.707f)
			{
				vec3 offset = their_pos - my_pos + their_vel * time;
				float sideDot = dot(offset, normalize(vec3(my_vel.y, -my_vel.x, 0)));
				steer = sideDot > 0 ? -1.0f : 1.0f;
			}
			else
			{
				if (parallelness > 0.707)
				{
					vec3 offset = their_pos - my_pos;
					float sideDot = dot(offset, normalize(vec3(my_vel.y, -my_vel.x, 0)));
					steer = sideDot > 0 ? -1.0f : 1.0f;
				}
				else
				{
					if (length(their_vel) <= length(my_vel))
					{ 
						float sideDot = dot(their_vel, normalize(vec3(my_vel.y, -my_vel.x, 0)));
						steer = sideDot > 0 ? -1.0f : 1.0f;
					}
				}
			}
		}
	}
	return steer * normalize(vec3(my_vel.y, -my_vel.x, 0) / time;
}

__device__
float predictNearestApproachTime(glm::vec3 my_pos, glm::vec3 my_vel, glm::vec3 their_pos, glm::vec3 their_vel)
{
	const glm::vec3 relVelocity = their_vel - my_vel;
	const float relSpeed = relVelocity.length();

	if (relSpeed == 0) return 0;

	const glm::vec3 relTangent = relVelocity / relSpeed;

	const glm::vec3 relPosition = my_pos - their_pos;
	const float projection = relTangent.x * relPosition.x + relTangent.y * relPosition.y + relTangent.z * relPosition.z;

	return projection / relSpeed;
}

__device__
float computeNearestApproachPositions(glm::vec3 my_pos, glm::vec3 my_vel, glm::vec3 their_pos, glm::vec3 their_vel, float time)
{
	return glm::distance(my_pos + my_vel * time, their_pos + their_vel * time);
}


__device__
bool inBoidNeighborhood(vec3 my_pos, vec3 my_vel, vec3 their_pos, vec3 their_vel, float minDist, float maxDist, float cosMaxAngle)
{
	vec3 offset = their_pos - my_pos;
	float dist = length(offset);
	if (dist < minDist)
		return true;
	else
	{
		if (dist > maxDist)
			return false;
		float forwardness = normalize(my_vel).dot(normalize(unitOffset));
		return forwardness > cosMaxAngle;
	}
}

__device__
vec3 steerToAvoidCloseNeighbor(vec3 my_pos, vec3 my_vel, vec3 their_pos, vec3 their_vel, float maxDist, float cosMaxAngle)
{
	if (inBoidNeighborhood(my_pos, my_vel, their_pos, their_vel, maxDist, cosMaxAngle)
	{
		vec3 offset - their_pos - my_pos;
		float dist = length(offset);
		return offset / (-dist * dist);
	}
	return vec3(0.0);
} 