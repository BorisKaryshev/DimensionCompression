#pragma once
#include <vector>
#include <iostream>
#include <string>

namespace MyCode {
	double Normalize(double x);

	class vector : public std::vector<double> {
	public:
		using std::vector<double>::vector;
		std::ostream& print(std::ostream& os);
		vector& operator -=(const vector& v);
		vector& operator *=(const double& v);
		vector& operator=(const std::vector<double> v);
	};

	vector operator *(vector& v, const double& a);
	std::vector<vector>& operator-=(std::vector<vector>& a, std::vector<vector>& b);
	std::vector<vector>& operator*=(std::vector<vector>& a, double& b);

	void Normalize(vector& v);

	double operator*(const vector& a, const vector& b);
	std::vector<vector>& operator*(std::vector<vector>& a, const double& b);

	struct Layer {
		vector values;
		vector* weights;
		int number_of_neurons;
	};

	class NeuralNetwork {
	protected:
		int number_of_layers;
		Layer* layers;
		virtual double ComputeValue(int layer_index, int neuron_index);
		void Null();
	public:
		NeuralNetwork() :layers{ nullptr }, number_of_layers{ 0 } {}
		NeuralNetwork(int n);
		NeuralNetwork(std::initializer_list<int> lst);
		~NeuralNetwork() {
			for (int i = 0; i < number_of_layers; i++)
				delete[] layers[i].weights;
		}

		void Train(std::vector<vector>& dataset, std::vector<vector>& answers, int number_of_iterations, double speed);
		int& NumberOfLayers() { return number_of_layers; }
		Layer& operator[](int i) { return layers[i]; }
		vector ComputeVector(vector& input, int layer_index_output);
	};


	vector CountError(vector& v1, vector& v2);

	double CountAverageError(const vector& v);
}

std::ostream& operator<<(std::ostream& os, MyCode::NeuralNetwork& NN);
std::ostream& operator<<(std::ostream& os, MyCode::vector& v);
std::ostream& operator<<(std::ostream& os, const std::vector<double> v);

std::vector<MyCode::vector> StringToVectorOfVectors(const std::string str, const char separator);