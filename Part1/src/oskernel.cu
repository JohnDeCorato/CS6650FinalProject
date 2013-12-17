#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "oskernel.h"
#include "utilities.h"
#include "const.h"
#include "device_functions.h"

using namespace glm;

// Utility Functions for vec3s for opensteer
__device__
vec3 truncateLength(vec3 inVec, float maxLength)
{
	float len = length(inVec);
	if (len <= maxLength)
		return inVec;
	else
		return inVec * maxLength / len;
}


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
		if (computeNearestApproachPositions(my_pos, my_vel, their_pos, their_vel, time))
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
	return steer * normalize(vec3(my_vel.y, -my_vel.x, 0)) / time;
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
		float forwardness = dot(normalize(my_vel), normalize(offset));
		return forwardness > cosMaxAngle;
	}
}

__device__
vec3 steerToAvoidCloseNeighbor(vec3 my_pos, vec3 my_vel, vec3 their_pos, vec3 their_vel, float maxDist, float cosMaxAngle)
{
	if (inBoidNeighborhood(my_pos, my_vel, their_pos, their_vel, 0.003, maxDist, cosMaxAngle))
	{
		vec3 offset = their_pos - my_pos;
		float dist = length(offset);
		return offset / (-dist * dist);
	}
	return vec3(0.0);
}

__device__
vec3 parallelComponent(const vec3 source, const vec3 unitBasis)
{
    const float projection = dot(source, unitBasis);
    return unitBasis * projection;
}

__device__
vec3 perpendicularComponent(const vec3 source, const vec3 unitBasis)
{
	return source - parallelComponent(source, unitBasis);
}

__device__
vec3 limitDeviationAngleUtility(const bool insideOrOutside,
                                          vec3& source,
                                          const float cosineOfConeAngle,
                                          vec3& basis)
{
    // immediately return zero length input vectors
    float sourceLength = length(source);
    if (sourceLength == 0) return source;

    // measure the angular diviation of "source" from "basis"
    vec3 direction = source / sourceLength;
    float cosineOfSourceAngle = dot(direction, basis);

    // Simply return "source" if it already meets the angle criteria.
    // (note: we hope this top "if" gets compiled out since the flag
    // is a constant when the function is inlined into its caller)
    if (insideOrOutside)
    {
		// source vector is already inside the cone, just return it
		if (cosineOfSourceAngle >= cosineOfConeAngle) return source;
    }
    else
    {
		// source vector is already outside the cone, just return it
		if (cosineOfSourceAngle <= cosineOfConeAngle) return source;
    }

    // find the portion of "source" that is perpendicular to "basis"
    const vec3 perp = perpendicularComponent(source, basis);

    // normalize that perpendicular
    const vec3 unitPerp = normalize(perp);

    // construct a new vector whose length equals the source vector,
    // and lies on the intersection of a plane (formed the source and
    // basis vectors) and a cone (whose axis is "basis" and whose
    // angle corresponds to cosineOfConeAngle)
    float perpDist = sqrt(1 - (cosineOfConeAngle * cosineOfConeAngle));
    vec3 c0 = basis * cosineOfConeAngle;
    vec3 c1 = unitPerp * perpDist;
    return (c0 + c1) * sourceLength;
}

__device__
vec3 limitMaxDeviationAngle(vec3& source, const float cosineOfConeAngle, vec3& basis)
{
	return limitDeviationAngleUtility(true, source, cosineOfConeAngle, basis);
}

__device__
vec3 limitMinDeviationAngle(vec3& source, const float cosineOfConeAngle, vec3& basis)
{
	return limitDeviationAngleUtility(false, source, cosineOfConeAngle, basis);
}

__device__
vec3 adjustRawSteeringForce(vec3 my_pos, vec3 my_vel, vec3 force)
{
	const float maxAdjustedSpeed = 0.2f * MAX_SPEED;
	if (length(my_vel) > maxAdjustedSpeed || force == vec3(0))
	{
		return force;
	}
	else 
	{
		const float range = length(my_vel) / maxAdjustedSpeed;
		const float cosine = 1.0 + -2.0 * pow(range, 20);
		return limitMaxDeviationAngle(force, cosine, normalize(my_vel));
	}
}

__device__
void applySteeringForce(vec3 my_pos, vec3 my_vel, vec3 force, float dt)
{
	vec3 adjustedForce = adjustRawSteeringForce(my_pos, my_vel, force);
	vec3 clippedForce = truncateLength(adjustedForce, MAX_FORCE);

	vec3 newAccel = clippedForce / AGENT_MASS;

	my_vel += newAccel * dt;
	my_pos += my_vel * dt; 
}

