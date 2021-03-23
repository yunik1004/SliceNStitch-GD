#pragma once

#include "utils.hpp"
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

class SpTensor {
public:
    SpTensor(const std::vector<int>& dimension);

    // Getters
    const std::vector<int>& dimension(void) const { return _dimension; }
    int numMode(void) const { return _numMode; }

protected:
    std::vector<int> _dimension;
    int _numMode;
};

class SpTensor_Hash : public SpTensor {
public:
    typedef std::unordered_map<std::vector<int>, double> coord_map;
    typedef std::vector<coord_map> row_vector;

    SpTensor_Hash(const std::vector<int>& dimension);

    double find(const std::vector<int>& coord) const;
    void insert(const std::vector<int>& coord, double value);
    void set(const std::vector<int>& coord, double value); // Override the existed value

    const std::vector<row_vector>& elems(void) const { return _elems; }

    void clear(void);
    const unsigned long long numNnz(void) const { return _numNnz; }

    double norm_frobenius(void) const;
    double norm_frobenius_latest(void) const;

protected:
    unsigned long long _numNnz = 0;
    std::vector<row_vector> _elems; // tree-like
};

class SpTensor_dX {
public:
    SpTensor_dX(void) {}

    double find(const std::vector<int>& coord) const;
    void insert(const std::vector<int>& coord, double value);

    const std::unordered_map<std::vector<int>, double>& elems(void) const { return _elems; }

    void clear(void);
    const unsigned long long numNnz(void) const { return _numNnz; }

private:
    std::unordered_map<std::vector<int>, double> _elems;
    unsigned long long _numNnz = 0;
};
