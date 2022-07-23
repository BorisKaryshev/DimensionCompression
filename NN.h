#ifndef MyNN
#define MyNN
#include "simple_matrix.h"
#include <functional>
#include <vector>

namespace MyCode
{

    template <typename T>
    struct Layer
    {
        std::function<T(T)> activation = [](T x)
        { return x; };
        std::function<T(T)> backpropagation = [](T x)
        { return x; };
        simple_matrix<T> values, weights;
        Layer(std::size_t number_of_neurons, std::size_t prev_number_of_neurons) : values(number_of_neurons, 1, [](const unsigned i, const unsigned j)
                                                                                          { return rand(); }), weights();
    }

    template <typename T>
    class NeuralNetwork
    {
    private:
        std::vector<Layer<T>> layers;

    public:
        NeuralNetwork(const std::size_t number_of_layers = 0, )
    };
}

#endif