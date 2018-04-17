/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  default_random_engine gen;
  
  // Create a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Set the number of particles to something for now
  num_particles = 100;
  
  for (int i = 0; i < num_particles; ++i) {
    
    Particle p;
    p.id = i;
    
    // Sample from the above normal distrubtions for each particle
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    
    particles.push_back(p);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  //When yaw_rate = 0; the final positions are calculated as:
  // xf = x0 + v.dt.(cos(yaw)) where cos(yaw) is the x component of the velocity
  // yf = y0 + v.dt.(sin(yaw)) where sin(yaw) is the y component of the velocity
  // yawf = yaw
  
  //when yaw_rate != 0; the final positions are calculated as:
  // xf = x0 + [v.(sin(yaw + yaw_rate.dt) - sin(yaw)) / yaw_rate] --> (1)
  // yf = y0 + [v.(cos(yaw) - cos(yaw + yaw_rate.dt)) / yaw_rate] --> (2)
  // yawf = yaw + yaw_rate.dt --> (3)
  
  // Using equations (1), (2) and (3) for predictiions:
  
  for (int i = 0; i < num_particles; ++i) {
    
    double cur_x = particles[i].x;
    double cur_y = particles[i].y;
    double cur_theta = particles[i].theta;
  
    //Predict x, y and theta
    double pred_x = cur_x;
    double pred_y = cur_y;
    double pred_theta = cur_theta;
    if(fabs(yaw_rate) < 0.0001) {
      pred_x = cur_x + velocity * delta_t * cos(cur_theta);
      pred_y = cur_y + velocity * delta_t * sin(cur_theta);
      pred_theta = cur_theta;
    }
    else {
      pred_x = cur_x + (velocity * delta_t * (sin(cur_theta + yaw_rate * delta_t) - sin(cur_theta)) / yaw_rate);
      pred_y = cur_y + (velocity * delta_t * (cos(cur_theta) - cos(cur_theta + yaw_rate * delta_t)) / yaw_rate);
      pred_theta = cur_theta + yaw_rate * delta_t;
    }
    
    default_random_engine gen;
    
    // Create a normal (Gaussian) distribution for predicted x, y and theta
    normal_distribution<double> dist_x(pred_x, std_pos[0]);
    normal_distribution<double> dist_y(pred_x, std_pos[1]);
    normal_distribution<double> dist_theta(pred_x, std_pos[2]);
    
    //Add above noise to the prediction and set
    particles[i].x = pred_x + dist_x(gen);
    particles[i].y = pred_y + dist_y(gen);
    particles[i].theta = pred_theta + dist_theta(gen);
    
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  
  for (auto &obs : observations) {
    
    // init min distance to a high value before we begin looping over all predicted landmark measurements
    double min_dist = numeric_limits<double>::max();
    
    for (auto &pred : predicted) {
      
      // calculate the distance b/w the current predicted landmark & the observed landmark measurement
      double obs_pred_dist = dist(pred.x, pred.y, obs.x, obs.y);
      
      // if this distance is less than anything we've seen so far for this observed landmark measurement
      // then update the association and the min_dist
      if(obs_pred_dist < min_dist) {
        min_dist = obs_pred_dist;
        obs.id = pred.id;
      }
    }
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  //pre-calculate the weight constants so we don't do it for each particle
  const double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  const double x_denom = 1 / (2 * pow(std_landmark[0], 2));
  const double y_denom = 1 / (2 * pow(std_landmark[1], 2));
  
  //Go over all the particles one by one
  for (auto &p : particles) {
    
    // this vector will hold all the map landmarks predicted to be in the sensor range of the cur particle
    std::vector<LandmarkObs> predicted_landmark;

    // go over all the map landmarks and find which ones fall in the sensor range for this particle
    int map_size = map_landmarks.landmark_list.size();
    for (int i = 0; i < map_size; ++i) {
      
      // find the distance b/w this landmark and the current particle
      double landmark_particle_dist = dist(p.x, p.y, map_landmarks.landmark_list[i].x_f,
                                           map_landmarks.landmark_list[i].y_f);
      
      // check if the above distance is in the sensor range and add to predicted if it is
      if( landmark_particle_dist < sensor_range)
        predicted_landmark.push_back(LandmarkObs {map_landmarks.landmark_list[i].id_i,
                                                  map_landmarks.landmark_list[i].x_f,
                                                  map_landmarks.landmark_list[i].y_f});
    }

    // this vector will hold the tranformed observations
    std::vector<LandmarkObs> observations_transformed;
    
    // transform the observations from the vehicle's to map's coordinate system
    for(auto obs : observations) {
      double obs_trans_x, obs_trans_y;
      obs_trans_x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
      obs_trans_y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
      observations_transformed.push_back(LandmarkObs{ obs.id, obs_trans_x, obs_trans_y });
    }

    // now make an association based on which of the predicted landmarks is closest to the
    // observed landmark from the current particle
    dataAssociation(predicted_landmark, observations_transformed);
    
    // now calculate the new weight of the particle as the product of each measurement's
    // Multivariate-Gaussian probability density

    // make sure the weight is set
    p.weight = 1;
    
    // go over all transformed observations and find the associated prediction
    // then calucate the weight
    for (auto obs : observations_transformed) {

      for (auto pred : predicted_landmark) {

        if (pred.id == obs.id) {

          double weight = a * exp( -( pow(pred.x - obs.x, 2) * x_denom) + (pow(pred.y - obs.y, 2) * y_denom));
          p.weight *= weight;
          break;
        }
      }
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  default_random_engine gen;
  
  // find the max weight and create a separate vector which has all the particle weights
  std::vector<double> weights;
  double max_weight = numeric_limits<double>::min();
  for (int i = 0; i < num_particles; ++i) {
    
    Particle p = particles[i];
    if (p.weight > max_weight)
      max_weight = p.weight;
    
    weights.push_back(p.weight);
  }

  //create a discrete distribution of the above weights to sample from
  std::uniform_int_distribution<int>  discrete_distr(0, num_particles);
  
  // get a particle index uniformly from the set of all indices in sampler
  int index = discrete_distr(gen);

  // now calculate the upper limit for the uniform continuous distribution
  double upper_limit = 2 * max_weight;
  std::uniform_real_distribution<double>  continuous_distr(0, upper_limit);

  // resample
  std::vector<Particle> resampled_particles;
  double beta = 0;
  for (int i = 0; i < num_particles; ++i) {
    beta += continuous_distr(gen);
    while (weights[index] < beta) {
      beta = beta - weights[index];
      index = (index + 1) % num_particles;
    }
    
    // add the selected index to particles
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
