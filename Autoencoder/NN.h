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
        std::function<T(T)> activation_derivative = [](T x)
        { return 1; };
        simple_matrix<T> values, weights;

        Layer(
            std::size_t number_of_neurons,
            std::size_t prev_number_of_neurons
        ) : values(
                number_of_neurons,
                1, 
                [](const unsigned i, const unsigned j){ return T{};}
            ), 
            weights(
                number_of_neurons,
                prev_number_of_neurons,
                [](const unsigned i, const unsigned j){ return ((int)rand()%10);}
            ) {};
    };

    template <typename T>
    class NeuralNetwork
    {
    public:
        std::vector<Layer<T>> layers;
        NeuralNetwork() = default;
        NeuralNetwork(const std::size_t number_of_layers) 
            : layers(number_of_layers) {};

        NeuralNetwork& add_layer(std::size_t number_of_neurons,
                                 std::function<T(T)> activation = [](T x){return x;},
                                 std::function<T(T)> activation_der = [](T x){return 1;}
        ) {
            if (layers.size() > 0){
                layers.push_back(
                    Layer<T> {number_of_neurons, 
                              layers[layers.size()-1].values.rows()
                    }
                );
            } else {
                layers.push_back(Layer<T> {number_of_neurons, 1});
            }
            layers[layers.size() - 1].activation = activation;
            layers[layers.size() - 1].activation_derivative = activation_der;
            
            return *this;
        }

        std::size_t number_of_layers() {return layers.size();};
        simple_matrix<T> 
        count_vector(simple_matrix<T> &data, std::size_t output_layer) {
#ifdef DEBUG
            if(data.columns() > 1) {
                throw std::out_of_range {"ERROR: wrong matrix size"};
            }
#endif
            layers[0].values = data;

            for(std::size_t i = 1; i < output_layer; i++) {
                auto prev_val = MyCode::for_each_matrix_copy(layers[i-1].values, 
                                                            layers[i-1].activation
                );
                layers[i].values =  layers[i].weights * prev_val;
            }
            return MyCode::for_each_matrix_copy(layers[output_layer-1].values, 
                                                layers[output_layer-1].activation
            );
        };

        double back_prop(
            simple_matrix<T> &values, 
            simple_matrix<T> &output, 
            std::size_t output_layer, 
            double speed = 1) 
        {
            (*this).count_vector(values, output_layer);
            auto error = MyCode::for_each_matrix_copy(layers[output_layer-1].values, 
                                                      layers[output_layer-1].activation
            ); 
            error = error - output;
            double average_error = 0;
            for(int i = 0; i < error.rows(); i++) {
                average_error += error[i][0]*error[i][0];
            }

            auto der_val = MyCode::for_each_matrix_copy(layers[output_layer-1].values, 
                                                        layers[output_layer-1].activation_derivative
            );
            auto d = MyCode::element_wise_multiply(error, der_val);
            for(size_t i = output_layer - 1; i > 0; i--) {
                auto delta = MyCode::for_each_matrix_copy(layers[i-1].values, 
                                                          layers[i-1].activation
                ); 
                delta.transpose();
                delta = d * delta;
                delta = delta * speed;
                der_val = layers[i].weights;
                layers[i].weights -= delta;

                der_val.transpose();
                d = der_val * d;
            }

            if(average_error != average_error)
                throw std::out_of_range {"ERROR: average error is NaN"};
            return average_error;
        }
    };
}

#endif