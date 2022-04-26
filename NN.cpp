#include <vector>
#include <iostream>
#include "NN.h"

namespace MyCode {
	std::ostream& vector::print(std::ostream& os) {
		for (auto& x : *this)
			os << x << " ";
		return os << "\n";
	}
	vector& vector::operator -=(const vector& v) {
		int i = 0;
		for (auto& x : *this)
			x -= v[i++];
		return *this;
	}
	vector& vector::operator *=(const double& v) {
		for (auto& x : *this)
			x *= v;
		return *this;
	}

	vector operator *(vector& v, const double& a) {
		vector temp;
		for (auto& x : v)
			temp.push_back(x * a);
		return temp;
	}
	std::vector<vector>& operator-=(std::vector<vector>& a, std::vector<vector>& b) {
		int i = 0;
		for (auto& x : a)
			x -= b[i++];
		return a;
	}
	std::vector<vector>& operator*=(std::vector<vector>& a, double& b) {
		for (auto& x : a)
			x *= b;
		return a;
	}
	vector& vector::operator=(const std::vector<double> v){
		for (auto i : v)
			(*this).push_back(i);

		return *this;
	}

	double operator*(const vector& a, const vector& b) {
		double rez = 0;
		if (a.size() != b.size())
			throw std::out_of_range{ "ERROR: trying to multiply vectors with different sizes\n" };
		for (int i = 0; i < a.size(); i++)
			rez += a[i] * b[i];
		return rez;
	}
	std::vector<vector>& operator*(std::vector<vector>& a, const double& b) {
		for (auto& x : a)
			x *= b;
		return a;
	}

	vector CountError(vector& v1, vector& v2) {
		if (v1.size() != v2.size()) {
			std::cout << "v1.size = " << v1.size() << " v2.size = " << v2.size() << '\n';
			throw std::out_of_range{ "ERROR: wrong vector size while counting error\n" };
		}
		vector output;
		int i = 0;
		for (int i = 0; i < v1.size(); i++)
			output.push_back(v1[i] - v2[i]);
		return output;
	}

	double CountAverageError(const vector& v) {
		double error = 0;
		for (auto& x : v)
			error += x * x;
		return error;
	}

	void NeuralNetwork::Train(std::vector<vector>& dataset, std::vector<vector>& answers, int number_of_iterations, double speed) {
		if (dataset.size() != answers.size())
			throw std::out_of_range{ "ERROR: dataset and answers size are not equal\n" };
		// Подсчёт ошибки
		for (int iteration = 0; iteration < number_of_iterations; iteration++) {
			double average_error = 0;
			for (int pos_in_dataset = 0; pos_in_dataset < dataset.size(); pos_in_dataset++) {
				ComputeVector(dataset[pos_in_dataset], number_of_layers - 1);
				for (int layer_pos = number_of_layers - 1; layer_pos > 0; layer_pos--) {
					vector error;
					if (layer_pos == number_of_layers - 1) {
						error = CountError(layers[number_of_layers - 1].values, answers[pos_in_dataset]);
						/*std::cout << error;
						_sleep(1000);*/
						average_error = (average_error*(pos_in_dataset) + CountAverageError(error))/(pos_in_dataset+1);
					}
					else
						error = layers[layer_pos].values;

					std::vector<vector> weights_adjusment(layers[layer_pos].number_of_neurons);
					for (auto& x : weights_adjusment)
						while (x.size() < layers[layer_pos - 1].number_of_neurons)
							x.push_back(0);

					for (int i = 0; i < layers[layer_pos - 1].number_of_neurons; i++)
						for (int j = 0; j < layers[layer_pos].number_of_neurons; j++) {
							weights_adjusment[j][i] = layers[layer_pos - 1].values[i] * error[j];
						}

					// Подсчёт корректировки весов				

					for (int i = 0; i < layers[layer_pos - 1].number_of_neurons; i++) {
						layers[layer_pos - 1].values[i] = 0;
						for (int j = 0; j < layers[layer_pos].number_of_neurons; j++) {
							layers[layer_pos - 1].values[i] += layers[layer_pos].weights[j][i] * error[j];
						}
					}
					// Передача ошибки на предыдущий слой

					weights_adjusment *= speed;
					for (int i = 0; i < layers[layer_pos].number_of_neurons; i++)
						layers[layer_pos].weights[i] -= weights_adjusment[i];
					// Корректировка весов на слое
				}
			}
			std::string s = "Iteration " + std::to_string(iteration + 1) + ", error = " + std::to_string(average_error) + "\n";
			std::cout << s;
		}

	}

	double NeuralNetwork::ComputeValue(int layer_index, int neuron_index) {

		if (layer_index < 0 || layer_index > number_of_layers)
			throw std::out_of_range{ "ERROR: wrong layer index while coumputing value\n" };
		if (neuron_index < 0 || neuron_index > layers[layer_index].number_of_neurons)
			throw std::out_of_range{ "ERROR: wrong neuron index while coumputing value\n" };

		if (layers[layer_index].values[neuron_index] != -1)
			return layers[layer_index].values[neuron_index];
		else {
			for (int i = 0; i < layers[layer_index - 1].number_of_neurons; i++)
				if (layers[layer_index - 1].values[i] == -1)
					layers[layer_index - 1].values[i] = ComputeValue(layer_index - 1, i);
			return layers[layer_index - 1].values * layers[layer_index].weights[neuron_index];
		}

	}

	vector NeuralNetwork::ComputeVector(vector& input, int layer_index_output) {
		Null();
		if (input.size() != layers[0].number_of_neurons)
			throw std::out_of_range{ "ERROR: invalid number of values in input vector\n" };

		layers[0].values = input;

		for (int i = 0; i < layers[layer_index_output].number_of_neurons; i++)
			layers[layer_index_output].values[i] = ComputeValue(layer_index_output, i);

		return layers[layer_index_output].values;
	}

	void NeuralNetwork::Null() {
		for (int i = 0; i < number_of_layers; i++)
			for (int j = 0; j < layers[i].number_of_neurons; j++)
				layers[i].values[j] = -1;
	}

	NeuralNetwork::NeuralNetwork(int n) :layers{ new Layer[1] }, number_of_layers{ 1 } {
		if (n <= 0)
			throw std::out_of_range{ "ERROR: invalid number of neurons\n" };
		layers[0].number_of_neurons = n;
		for (int i = 0; i < n; i++)
			layers[0].values.push_back(-1);
		layers[0].weights = nullptr;
	}

	NeuralNetwork::NeuralNetwork(std::initializer_list<int> lst) :layers{ new Layer[lst.size()] }, number_of_layers{ (int)lst.size() } {
		int i = 0;
		for (auto& x : lst) {
			if (x <= 0)
				throw std::out_of_range{ "ERROR: invalid number of neurons\n" };
			layers[i].number_of_neurons = x;
			for (int j = 0; j < x; j++)
				layers[i].values.push_back(-1);
			if (i == 0) {
				layers[i].weights = nullptr;
			}
			else {
				layers[i].weights = new vector[x];
				for (int j = 0; j < layers[i].number_of_neurons; j++) {
					for (int k = 0; k < layers[i - 1].number_of_neurons; k++)
						layers[i].weights[j].push_back(rand() % 3 - 1);
				}
			}
			i++;
		}
	}

}

std::ostream& operator<<(std::ostream& os, MyCode::NeuralNetwork& NN) {
	if (NN.NumberOfLayers() <= 0)
		return os;
	NN[0].values.print(os);
	os << "---------\n";
	for (int i = 1; i < NN.NumberOfLayers(); i++) {
		NN[i].values.print(os);
		for (int j = 0; j < NN[i].number_of_neurons; j++)
			NN[i].weights[j].print(os);
		if (i != NN.NumberOfLayers() - 1)
			os << "---------\n";
	}
	return os;
}

std::ostream& operator<<(std::ostream& os, MyCode::vector& v) {
	for (auto& x : v)
		os << x << " ";
	return os << "\n";
}

std::vector<MyCode::vector> StringToVectorOfVectors(const std::string str, const char separator) {
	MyCode::vector v;
	std::vector<MyCode::vector> output;
	std::string buff = "";
	std::cout << "||" << str << "||\n";
	for (const char i : str) {
		if (i == separator) {
			v.push_back(std::stod(buff));
			buff = "";
		}
		else if (i == '\n') {
			v.push_back(std::stod(buff));
			buff = "";
			output.push_back(v);
			v.erase(v.begin(), v.end());
		} else
			buff += i;
	}

	return output;
}

std::ostream& operator<<(std::ostream& os, const std::vector<double> v) {
	for (auto i : v)
		os << i << ", ";
	os << "\n";
	return os;
}