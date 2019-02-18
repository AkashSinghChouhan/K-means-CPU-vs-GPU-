#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <stdio.h>

#include <GLUT/glut.h>
#include <GL/gl.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int k = 8;
const auto number_of_iterations = 300;

struct Data {

	float* x{ nullptr };
	float* y{ nullptr };
	float* z{ nullptr };
	int* label{ nullptr };
	int size{ 0 };
	int bytes{ 0 };
	

	Data() {}

	Data(int s) : size(s), bytes(size * sizeof(float)) {
		cudaMalloc(&x, bytes);
		cudaMalloc(&y, bytes);
		cudaMalloc(&z, bytes);
		cudaMalloc(&label, bytes);
		
	}

	Data(int s, std::vector<float>& h_x, std::vector<float>& h_y, std::vector<float>& h_z, std::vector<int>& h_label)
		: size(s), bytes(size * sizeof(float)) {
		cudaMalloc(&x, bytes);
		cudaMalloc(&y, bytes);
		cudaMalloc(&z, bytes);
		cudaMalloc(&label, bytes);
		
		cudaMemcpy(x, h_x.data(), bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(y, h_y.data(), bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(z, h_z.data(), bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(label, h_label.data(), bytes, cudaMemcpyHostToDevice);
		
		
	}
	

	~Data() {
		cudaFree(x);
		cudaFree(y);
		cudaFree(z);
		cudaFree(label);
		
	}

	
	void clear() {
		cudaMemset(x, 0, bytes);
		cudaMemset(y, 0, bytes);
		cudaMemset(z, 0 ,bytes);
		cudaMemset(label, 0, bytes);
		
	
	}

};

__device__ float
squared_l2_distance(float x_1, float y_1, float z_1, float x_2, float y_2, float z_2) {
	return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2) + (z_1 - z_2)*(z_1 - z_2) ;
}
float h_squared_l2_distance(float x_1, float y_1, float z_1, float x_2, float y_2, float z_2) {
	return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2) + (z_1 - z_2)*(z_1 - z_2);
}

__global__ void assign_clusters(const float* __restrict__ data_x,
	const float* __restrict__ data_y,
	const float* __restrict__ data_z,
	int data_size,
	const float* __restrict__ means_x,
	const float* __restrict__ means_y,
	const float* __restrict__ means_z,
	float* __restrict__ new_sums_x,
	float* __restrict__ new_sums_y,
	float* __restrict__ new_sums_z,
	int k,
	int* __restrict__ counts
	) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= data_size) return;

	// Make global loads once.
	const float x = data_x[index];
	const float y = data_y[index];
	const float z = data_z[index];

	float best_distance = FLT_MAX;
	int best_cluster = 0;
	for (int cluster = 0; cluster < k; ++cluster) {
		const float distance =
			squared_l2_distance(x, y, z, means_x[cluster], means_y[cluster],means_z[cluster]);
		if (distance < best_distance) {
			best_distance = distance;
			best_cluster = cluster;
		}
		
	}

	atomicAdd(&new_sums_x[best_cluster], x);
	atomicAdd(&new_sums_y[best_cluster], y);
	atomicAdd(&new_sums_z[best_cluster], z);
	atomicAdd(&counts[best_cluster], 1);
}

__global__ void compute_new_means(float* __restrict__ means_x,
	float* __restrict__ means_y,
	float* __restrict__ means_z,
	const float* __restrict__ new_sum_x,
	const float* __restrict__ new_sum_y,
	const float* __restrict__ new_sum_z,
	const int* __restrict__ counts
	) {
	
	const int cluster = threadIdx.x;
	const int count = max(1, counts[cluster]);
	means_x[cluster] = new_sum_x[cluster] / count;
	means_y[cluster] = new_sum_y[cluster] / count;
	means_z[cluster] = new_sum_z[cluster] / count;
}

void assign_label(std::vector<float> means_x,
	std::vector<float> means_y,
	std::vector<float> means_z,
	std::vector<int> &means_label,
	std::vector<float> data_x,
	std::vector<float> data_y,
	std::vector<float> data_z,
	std::vector<int> &data_label,
	int k,
	int number_of_elements) {

	for (int i = 0; i < k ; i++) { means_label.at(i)=i+1; }
	float distance;
	int label;
	//Assign label.....   
	for (int i = 0; i < number_of_elements; i++) {

		
		auto distanceMin = FLT_MAX;
		for (int j = 0; j < k; j++) {

			distance = h_squared_l2_distance(means_x.at(j),means_y.at(j),means_z.at(j),data_x.at(i),data_y.at(i),data_z.at(i));

			if (distance < distanceMin) { distanceMin = distance; label = means_label.at(j); }


		}
		data_label.at(i) = label;


	}


}
