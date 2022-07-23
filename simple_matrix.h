#ifndef SIMPLE_MATRIX
#define SIMPLE_MATRIX
#include <vector>
#include <iostream>
#include <functional>

template <typename T>
class simple_matrix {
    private:
        std::vector<std::vector<T>> data;
    public:
        simple_matrix(const unsigned rows = 0, const unsigned columns = 0, std::function<T (const unsigned, const unsigned)> op = [](const unsigned row, const unsigned column) {
            return 0;
        }) : data(rows, std::vector<T> (columns)) {
            for(int i = 0; i < rows; i++)
                for(int j = 0; j < columns; j++)
                    data[i][j] = op(i, j);
        };
        simple_matrix(simple_matrix &m)  :data(m.rows(), std::vector<T> (m.columns())) {
            std::copy(m.data.begin(), m.data.end(), (*this).data.begin());
        }
        simple_matrix(simple_matrix &&m)  :data(m.rows(), std::vector<T> (m.columns())) {
            data = std::move(m.data);
        }

        simple_matrix& operator=(simple_matrix &m) {
            data = m.data;
            return *this;            
        }

        simple_matrix& operator=(simple_matrix &&m) {
            data = std::move(m.data);
            return *this;            
        }

        simple_matrix& operator+=(simple_matrix &m) {
            #ifdef DEBUG
            if ((*this).rows() != m.rows() || (*this).columns() != m.columns())
                throw std::out_of_range {R"(ERROR: wrong matrixies size on operation +=)"};
            #endif
            for(int i = 0; i < (*this).rows(); i++)
                for(int j = 0; j < (*this).columns(); j++)
                    data[i][j] += m.data[i][j];
            
            return *this;
        }

        simple_matrix& operator-=(simple_matrix &m) {
            #ifdef DEBUG
            if ((*this).rows() != m.rows() || (*this).columns() != m.columns())
                throw std::out_of_range {R"(ERROR: wrong matrixies size on operation -=)"};
            #endif
            for(int i = 0; i < (*this).rows(); i++)
                for(int j = 0; j < (*this).columns(); j++)
                    data[i][j] -= m.data[i][j];
            
            return *this;
        }

        ~simple_matrix() {};

        std::vector<T>& operator[](const unsigned i) {return data[i];}
        std::size_t rows() {return data.size();};
        std::size_t columns() {return data[0].size();};
        
};

template <typename T, typename U>
simple_matrix<decltype(T {} * U {})> operator*(simple_matrix<T> &a, simple_matrix<U> &b) {
    #ifdef DEBUG
    if(a.columns() != b.rows())
        throw std::out_of_range {"ERROR: wrong matrixies size on operation *"};
    #endif
    simple_matrix<decltype(T{} * U{})> out (a.rows(), b.columns(), [](const unsigned i, const unsigned j) {return decltype(T {} * U {}) {};});

    const auto rows = out.rows();
    const auto columns = out.columns();
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < columns; j++) {
            for(int k = 0; k < a.columns(); k++)
                out[i][j] += (a[i][k])*(b[k][j]);
        }

    return std::move(out);
}

template <typename T, typename Rand>
simple_matrix<T>& MakeMatrixRand(simple_matrix<T> &m, Rand rand) {
    for(int i = 0; i < m.rows(); i++)
        for(int j = 0; j < m.columns(); j++)
            m[i][j] = rand();
    
    return m;
}

template <typename T>
std::ostream& operator<<(std::ostream &os, std::vector<T> &v);

template <class T>
std::ostream& operator<<(std::ostream &os, simple_matrix<T> &m) {
    if(m.rows() == 0)
        return os;
    for(int i = 0; i < m.rows()-1; i++)
        os << m[i] << '\n';
    os << m[m.rows()-1];

    return os;
}

template <class T>
std::ostream& operator<<(std::ostream &os, std::vector<T> &v) {
    if(v.size() <= 0)
        return os;
    for(int i = 0; i < v.size()-1; i++)
        os << v[i] << ' ';
    os << v[v.size()-1];

    return os;
}

#endif
