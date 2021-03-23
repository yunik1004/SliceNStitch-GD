#include "tensor.hpp"
#include "constants.hpp"
#include <cmath>
#include <iterator>
#include <sstream>
#include <string>

SpTensor::SpTensor(const std::vector<int>& dimension)
{
    _dimension = dimension;
    _numMode = _dimension.size();
}

SpTensor_Hash::SpTensor_Hash(const std::vector<int>& dimension)
    : SpTensor(dimension)
{
    _elems.resize(_numMode);
    for (int i = 0; i < _numMode; ++i) {
        _elems[i].resize(_dimension[i]);
    }
}

double SpTensor_Hash::find(const std::vector<int>& coord) const
{
    const coord_map& cmap = _elems[0][coord[0]];
    const coord_map::const_iterator& it = cmap.find(coord);

    return (it != cmap.end()) ? it->second : 0;
}

void SpTensor_Hash::insert(const std::vector<int>& coord, double value)
{
    const double preValue = find(coord);
    const double newValue = preValue + value;
    const bool prevZero = abs(preValue) < TENSOR_MACHINE_EPSILON;
    const bool newZero = abs(newValue) < TENSOR_MACHINE_EPSILON;

    if (abs(newValue) < TENSOR_MACHINE_EPSILON) { // Treated as zero
        if (!prevZero) {
            for (int m = 0; m < _numMode; ++m) {
                _elems[m][coord[m]].erase(coord);
            }
            _numNnz--;
        }
    } else {
        for (int m = 0; m < _numMode; ++m) {
            _elems[m][coord[m]][coord] = newValue;
        }

        if (prevZero)
            _numNnz++;
    }
}

void SpTensor_Hash::set(const std::vector<int>& coord, double value)
{
    if (abs(value) < TENSOR_MACHINE_EPSILON) { // Treated as zero
        for (int m = 0; m < _numMode; ++m) {
            _elems[m][coord[m]].erase(coord);
        }
    } else {
        for (int m = 0; m < _numMode; ++m) {
            _elems[m][coord[m]][coord] = value;
        }
    }
}

void SpTensor_Hash::clear(void)
{
    _numNnz = 0;
    for (int m = 0; m < _numMode; ++m) {
        for (int i = 0; i < _dimension[m]; ++i) {
            _elems[m][i].clear();
        }
    }
}

double SpTensor_Hash::norm_frobenius(void) const
{
    double square_sum = 0;

    const row_vector& elems_vec = _elems[0];
    for (int i = 0; i < _dimension[0]; ++i) {
        for (auto const& it : elems_vec[i]) {
            const double val = it.second;
            square_sum += pow(val, 2);
        }
    }

    return sqrt(square_sum);
}

double SpTensor_Hash::norm_frobenius_latest(void) const
{
    double square_sum = 0;

    const row_vector& elems_vec = _elems[_numMode - 1];
    for (auto const& it : elems_vec[_dimension[_numMode - 1] - 1]) {
        const double val = it.second;
        square_sum += pow(val, 2);
    }

    return sqrt(square_sum);
}

double SpTensor_dX::find(const std::vector<int>& coord) const
{
    const std::unordered_map<std::vector<int>, double>::const_iterator& it = _elems.find(coord);

    return (it != _elems.end()) ? it->second : 0;
}

void SpTensor_dX::insert(const std::vector<int>& coord, double value)
{
    const double preValue = find(coord);
    const double newValue = preValue + value;

    const bool prevZero = abs(preValue) < TENSOR_MACHINE_EPSILON;
    const bool newZero = abs(newValue) < TENSOR_MACHINE_EPSILON;

    if (!prevZero && newZero) { // Treated as zero
        _elems.erase(coord);

        _numNnz--;
    } else if (!newZero) {
        _elems[coord] = newValue;

        if (prevZero) {
            _numNnz++;
        }
    }
}

void SpTensor_dX::clear(void)
{
    _elems.clear();
}
