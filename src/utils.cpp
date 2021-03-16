#include "utils.hpp"

RNG* RNG::_instance = nullptr;
bool RNG::_isRandom = true;

RNG::RNG(void)
{
    if (_isRandom) {
        _rd = new std::random_device();
        _rng = new std::mt19937((*_rd)());
    } else {
        _rng = new std::mt19937(0);
    }
}

RNG::~RNG(void)
{
    delete _rd;
    delete _rng;
}

void pickIdx_replacement(const std::vector<int>& dimension, int k, std::unordered_set<std::vector<int>>& sampledIdx)
{
    std::mt19937* rng = RNG::Instance()->rng();
    const int ndim = dimension.size();

    std::vector<int> v(ndim);
    for (int i = 0; i < k; ++i) {
        for (int m = 0; m < ndim; ++m) {
            v[m] = std::uniform_int_distribution<>(0, dimension[m] - 1)(*rng);
        }
        sampledIdx.insert(v);
    }
}
