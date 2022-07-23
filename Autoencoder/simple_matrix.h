#ifndef SIMPLE_MATRIX
#define SIMPLE_MATRIX
#include <vector>
#include <iostream>
#include <functional>
#include <type_traits>

namespace MyCode
{
    template <typename T>
    class simple_matrix;

    template <typename T, typename U>
    simple_matrix<decltype(T{} * U{})>
    operator*(simple_matrix<T> &, simple_matrix<U> &);

    template <typename T>
    class simple_matrix
    {
    private:
        std::vector<std::vector<T>> data;

    public:
        simple_matrix(
            std::size_t rows = 0,
            std::size_t columns = 0,
            std::function<T(const unsigned, const unsigned)> op 
                = [](std::size_t row, std::size_t column) { return T{}; }
        )
            : data(rows, std::vector<T>(columns))
        {
#ifdef DEBUG
            std::cout << "|Matrix created in functional way|\n";
#endif
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < columns; j++)
                    data[i][j] = op(i, j);
        };
        simple_matrix(simple_matrix &m) = default;
        simple_matrix(simple_matrix &&m) = default;

        simple_matrix &operator=(simple_matrix &m) = default;
        simple_matrix &operator=(simple_matrix &&m) = default;

        simple_matrix &operator+=(simple_matrix &m)
        {
            if ((*this).rows() != m.rows() 
                || (*this).columns() != m.columns())
            {
                throw std::out_of_range{
                    "ERROR: wrong matrixies size"
                    " on operation +="};
            }
            for (int i = 0; i < (*this).rows(); i++)
                for (int j = 0; j < (*this).columns(); j++)
                    data[i][j] += m.data[i][j];

            return *this;
        }

        simple_matrix &operator-=(simple_matrix &m)
        {
            if ((*this).rows() != m.rows() 
                || (*this).columns() != m.columns())
            {
                throw std::out_of_range{
                    "ERROR: wrong matrixies size"
                    " on operation -="};
            }
            for (int i = 0; i < (*this).rows(); i++)
                for (int j = 0; j < (*this).columns(); j++)
                    data[i][j] -= m.data[i][j];

            return *this;
        }
        simple_matrix &operator*=(simple_matrix &m)
        {
            (*this) = (*this) * m;
            return *this;
        }

        ~simple_matrix(){};

        std::vector<T> &operator[](const unsigned i) { return data[i]; }
        std::size_t rows() { return data.size(); };
        std::size_t columns() { return data[0].size(); };
        simple_matrix<T> transpose() {
            if (data.size() == data[0].size()) {
                for(size_t i = 0; i < data.size(); i++) {
                    for(size_t j = i + 1; j < data[0].size(); j++) {
                        std::swap(data[i][j], data[j][i]);
                    }
                }
            } else {
                std::vector<std::vector<T>> new_data (data[0].size(), 
                                                      std::vector<T> (data.size())
                );
                for(std::size_t i = 0; i < data.size(); i++) {
                    for(std::size_t j = 0 ; j < data[0].size(); j++) {
                        new_data[j][i] = data[i][j];
                    }
                }
                data = std::move(new_data);
            }
            return *this;
        }

    };

    template <typename T, typename U>
    inline simple_matrix<decltype(T{} * U{})>
    operator*(simple_matrix<T> &a, U b)
    {
        return simple_matrix<decltype(T{} * U{})>{
            a.rows(),
            a.columns(),
            [&](std::size_t i, std::size_t j) 
            { return a[i][j] * b; }
        };
    };

    template <typename T, typename U>
    inline simple_matrix<decltype(T{} * U{})> 
    operator*(simple_matrix<T> &a, simple_matrix<U> &b)
    {
        if (a.columns() != b.rows())
            throw std::out_of_range{
                "ERROR: wrong matrixies size"
                " on operation *"
        };
        simple_matrix<decltype(T{} * U{})> out(a.rows(),
                                               b.columns(),
                                               [](const unsigned i, const unsigned j)
                                               { return decltype(T{} * U{}){}; }
        );

        const auto rows = out.rows();
        const auto columns = out.columns();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) {
                for (int k = 0; k < a.columns(); k++)
                    out[i][j] += (a[i][k]) * (b[k][j]);
            }

        return out;
    }

    template <typename T, typename U>
    inline simple_matrix<decltype(T{} + U{})> 
    operator+(simple_matrix<T> &a, simple_matrix<U> &b)
    {
        if (a.rows() != b.rows() 
            or a.columns() != b.columns())
            throw std::out_of_range{
                "ERROR: wrong matrixies size"
                " on operation +"
            };
        return std::move(
            simple_matrix<decltype(T{} + U{})>(a.rows(), 
                                               a.columns(), 
                                               [&](const unsigned i, const unsigned j)
                                               { return a[i][j] + b[i][j]; }
            )
        );
    }

    template <typename T, typename U>
    inline simple_matrix<decltype(T{} - U{})> 
    operator-(simple_matrix<T> &a, simple_matrix<U> &b)
    {
        if (a.rows() != b.rows() 
            or a.columns() != b.columns())
            throw std::out_of_range{
                "ERROR: wrong matrixies size"
                " on operation +"};
        return std::move(
            simple_matrix<decltype(T{} - U{})>(a.rows(), 
                                               a.columns(), 
                                               [&](const unsigned i, const unsigned j)
                                               { return a[i][j] - b[i][j]; }
            )
        );
    }

    template <typename T, typename Rand>
    inline simple_matrix<T> &MakeMatrixRand(simple_matrix<T> &m, Rand rand)
    {
        for (int i = 0; i < m.rows(); i++)
            for (int j = 0; j < m.columns(); j++)
                m[i][j] = rand();

        return m;
    }

    template <typename T, typename U>
    simple_matrix<decltype(T {} * U {})>
    element_wise_multiply(simple_matrix<T> &a, simple_matrix<U> &b) {
        if(a.rows() != b.rows() || a.columns() != b.columns()) {
            throw std::out_of_range {"ERROR: wrong matrixies size on element wise multylication"};
        }
        return simple_matrix<decltype(T {} * U {})> {a.rows(), 
                                                     a.columns(), 
                                                     [&](const unsigned i, const unsigned j) 
                                                     {return a[i][j] * b[i][j];}
        };
    }

    template <typename T, typename OP>
    simple_matrix<T>
    for_each_matrix_copy(simple_matrix<T> &a, OP o) {
        return simple_matrix<T> {a.rows(), 
                                 a.columns(), 
                                 [&](const unsigned i, const unsigned j) 
                                 {return o(a[i][j]);}
        };
    }
}

template <typename T>
std::ostream &
operator<<(std::ostream &os, std::vector<T> &v);

template <class T>
std::ostream &
operator<<(std::ostream &os, MyCode::simple_matrix<T> &m)
{
    if (m.rows() == 0)
        return os;
    for (int i = 0; i < m.rows() - 1; i++)
        os << m[i] << '\n';
    os << m[m.rows() - 1];

    return os;
}

template <class T>
std::ostream &
operator<<(std::ostream &os, std::vector<T> &v)
{
    if (v.size() <= 0)
        return os;
    for (int i = 0; i < v.size() - 1; i++)
        os << v[i] << ' ';
    os << v[v.size() - 1];

    return os;
}

#endif
