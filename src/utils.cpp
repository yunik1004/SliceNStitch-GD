#include "utils.hpp"
#include "tensor.hpp"

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

StreamMiner::StreamMiner(int size, double prob)
    : _size(size)
    , _prob(prob)
{
    _storage.resize(_size);
}

void StreamMiner::initialize(SpTensor_Hash* X)
{
    const int numMode = X->numMode();
    const std::vector<SpTensor_Hash::coord_map>& elems = X->elems()[numMode - 1];

    int idx = 0;
    for (auto const& coord_map : elems) {
        for (auto const& it : coord_map) {
            const std::vector<int>& coord_vec = it.first;

            if (idx < _size) {
                _storage[idx++] = coord_vec;
            } else {
                insert(coord_vec);
            }
        }
    }
}

void StreamMiner::sample(int k, std::unordered_set<std::vector<int>>& sampledIdx) const
{
    std::mt19937* rng = RNG::Instance()->rng();
    std::uniform_int_distribution<int> dist(0, _size - 1);

    int idx;
    for (int i = 0; i < k; ++i) {
        idx = dist(*rng);
        sampledIdx.insert(_storage[idx]);
    }
}

void StreamMiner::insert(const std::vector<int>& index)
{
    std::mt19937* rng = RNG::Instance()->rng();
    const bool change = std::discrete_distribution<>({ 1 - _prob, _prob })(*rng) > 0.5;
    if (change) {
        const int idx = std::uniform_int_distribution<>(0, _size - 1)(*rng);
        _storage[idx] = index;
    }
}
