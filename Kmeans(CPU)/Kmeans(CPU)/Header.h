#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>



#include <GL\glut.h>
#include <GL/gl.h>

auto k = 5;
const auto iterations = 400;
const auto number_of_runs = 4;

struct Point {
	float x{ 0 }, y{ 0 }, z{ 0 };
public:
	int label;
	float getX() { return this->x; }
	float getY() { return this->y; }
	float getZ() { return this->z; }
};

using DataFrame = std::vector<Point>;
DataFrame means;

float square(float value) {
	return value * value;
}

float squared_l2_distance(Point first, Point second) {
	return square(first.x - second.x) + square(first.y - second.y) + square(first.z - second.z);
}

DataFrame k_means(  DataFrame& dataPoints,
	int k,
	int number_of_iterations) {
	static std::random_device seed;
	static std::mt19937 random_number_generator(seed());
	std::uniform_int_distribution<size_t> indices(0, dataPoints.size() - 1);

	// Pick centroids as random points from the dataset.
	//DataFrame means(k);
	for (auto& cluster : means) {
		cluster = dataPoints[indices(random_number_generator)];
	}

	std::vector<size_t> assignments(dataPoints.size());
	for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
		// Find assignments.
		for (size_t point = 0; point < dataPoints.size(); ++point) {
			auto best_distance = std::numeric_limits<float>::max();
			size_t best_cluster = 0;
			for (size_t cluster = 0; cluster < k; ++cluster) {
				const float distance =
					squared_l2_distance(dataPoints[point], means[cluster]);
				if (distance < best_distance) {
					best_distance = distance;
					best_cluster = cluster;
				}
			}
			assignments[point] = best_cluster;
		}

		// Sum up and count points for each cluster.
		DataFrame new_means(k);
		std::vector<size_t> counts(k, 0);
		for (size_t point = 0; point < dataPoints.size(); ++point) {
			const auto cluster = assignments[point];
			new_means[cluster].x += dataPoints[point].x;
			new_means[cluster].y += dataPoints[point].y;
			new_means[cluster].z += dataPoints[point].z;
			counts[cluster] += 1;
		}

		// Divide sums by counts to get new centroids.
		for (size_t cluster = 0; cluster < k; ++cluster) {
			// Turn 0/0 into 0/1 to avoid zero division.
			const auto count = std::max<size_t>(1, counts[cluster]);
			means[cluster].x = new_means[cluster].x / count;
			means[cluster].y = new_means[cluster].y / count;
			means[cluster].z = new_means[cluster].z / count;
			
		}
		for (int i = 0; i < means.size(); i++) { means[i].label = i ; }
		float distance;
		int label;
		//Assign label.....   can also use assignments for assigning labels
		for (int i = 0; i < dataPoints.size(); i++) {

			dataPoints[i].label = assignments[i];
			/*auto distanceMin = 1000000000;
			for (int j = 0; j < means.size(); j++) {

				distance = squared_l2_distance(means[j],dataPoints[i]);

				if (distance < distanceMin) { distanceMin = distance; label = means[j].label; }
				

			}
			dataPoints[i].label = label;*/
		
		
		}
	}

	return means;
}

