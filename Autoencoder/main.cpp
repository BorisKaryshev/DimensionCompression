#include <iostream>
#include <cmath>
#include "simple_matrix.h"
#include "NN.h"
using std::cout;

using matrix = MyCode::simple_matrix<double>;

struct RandomNumber {
        static bool exist; 
        RandomNumber() {
            if (!exist) {
                #ifdef DEBUG
                cout << "SRAND set\n";
                #endif
                srand(time(NULL));
                exist = true;
            }
        };
        double operator()() {return rand()%10;};
};
bool RandomNumber::exist = false;

template <typename T>
std::ostream& operator<<(std::ostream &os, MyCode::NeuralNetwork<T> &NN) {
    os << "\n----------\n";
    for(int i = 1 ; i < NN.number_of_layers(); i++) {
        os << NN.layers[i].weights << "\n----------\n";
    }
    return os;
}

#define SIZE 6
#define ARRAY_SIZE 3

void init(MyCode::simple_matrix<double> &m, double a, double b, double c) {
    m[0][0] = a;
    m[1][0] = b;
    m[2][0] = c;
}

void print(matrix *in, MyCode::NeuralNetwork<double> &NN) {
    for(size_t i = 0 ; i < SIZE; i++) {
        auto m = NN.count_vector(in[i], NN.number_of_layers());
        in[i].transpose();
        cout << in[i] << " = " << m << "\n";
        in[i].transpose();
    }
}

int main(){
    try {
        RandomNumber random;

        auto sigmoid = [](double x) {
            return (x > 0);// ? x : 0;
        };

        auto sigmoid_der = [](double x) {
            return  (x > 0)? 1 : 0;
        };

        matrix m[SIZE];
        m[0] = MyCode::simple_matrix<double> {3, 1};
        init(m[0], 1, 0, 1);
        m[1] = MyCode::simple_matrix<double> {3, 1};
        init(m[1], 0, 1, 1);
        m[2] = MyCode::simple_matrix<double> {3, 1};
        init(m[2], 0, 0, 1);
        m[3] = MyCode::simple_matrix<double> {3, 1};
        init(m[3], 1, 1, 1);
        m[4] = MyCode::simple_matrix<double> {3, 1};
        init(m[4], 0, 1, 1);
        m[5] = MyCode::simple_matrix<double> {3, 1};
        init(m[5], 1, 1, 1);

        matrix out[SIZE];
        out[0] = MyCode::simple_matrix<double> {1, 1};
        out[0][0][0] = 1;
        out[1] = MyCode::simple_matrix<double> {1, 1};
        out[1][0][0] = 1;
        out[2] = MyCode::simple_matrix<double> {1, 1};
        out[2][0][0] = 0;
        out[3] = MyCode::simple_matrix<double> {1, 1};
        out[3][0][0] = 0;
        out[4] = MyCode::simple_matrix<double> {1, 1};
        out[4][0][0] = 1;
        out[5] = MyCode::simple_matrix<double> {1, 1};
        out[5][0][0] = 0;

        MyCode::NeuralNetwork<double> NN;
        NN.add_layer(ARRAY_SIZE, sigmoid, sigmoid_der);
        NN.add_layer(ARRAY_SIZE*2);//, sigmoid, sigmoid_der);
        NN.add_layer(ARRAY_SIZE*2, sigmoid, sigmoid_der);
        NN.add_layer(1, sigmoid, sigmoid_der);

        print(m, NN);
        for(size_t iteration = 0; iteration < 10; iteration++) {
            double average_error = 0;
            for(size_t i = 0; i < SIZE; i++) {

                average_error = average_error * (i) + NN.back_prop(m[i], out[i], 4, 1000);
                average_error /= (i+1);
            }
            cout << "Iteration " << iteration + 1 << " average error = " << average_error << "\n";

        }
        print(m, NN);

    } catch( std::exception &ex) {
        cout << ex.what() << '\n';
        return -1;
    }
    

    return 0;
}
