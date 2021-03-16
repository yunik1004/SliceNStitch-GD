#pragma once

#include "loader.hpp"
#include <Eigen/Dense>
#include <any>
#include <chrono>
#include <map>
#include <random>
#include <unordered_set>
#include <vector>

class Config;
class SpTensor_List;
class SpTensor_Hash;
class SpTensor_dX;
class DataStream;

class TensorStream;
TensorStream* generateTensorStream(DataStream& paperX, const Config& config); // Generate the tensor stream

class TensorStream {
public:
    TensorStream(DataStream& paperX, const Config& config);
    virtual ~TensorStream(void);

    void updateTensor(const DataStream::Event& e);
    void updateFactor(void);

    double elapsedTime(void) const; // sec

    double find_reconst(const std::vector<int>& coord) const;
    double density(void) const;

    /* Load matrix */
    void saveFactor(std::string fileName) const;

    /* Get errors */
    double rmse(void) const;
    double fitness(void) const;
    double fitness_latest(void) const;
    double error(const std::vector<int>& coord) const; // Error of the given entry

    void updateAtA(void); // Update the AtA when _use_AtA is false

protected:
    const Config* _config;

    std::vector<int> _compute_order;

    virtual void _updateAlgorithm(void) {} // It will change the current updateAlgorithm later

    double _norm_frobenius_reconst(void) const;
    double _innerprod_X_X_reconst(void) const;

    SpTensor_Hash* _X = nullptr;
    SpTensor_dX* _dX = nullptr;
    DataStream* _paperX;

    Eigen::ArrayXd _lambda;
    std::vector<Eigen::MatrixXd> _A;
    std::vector<Eigen::ArrayXXd> _AtA;

    bool _use_AtA = true;

    long long _nextTime;

    std::chrono::nanoseconds _elapsed_time; // Elapsed time

    void _rand_init_A(void); // Randomly initialize factor matrices

    void _als_base(void); // Base code for ALS
    void _unnormalize_A(void); // Unnormalize the factor matrices

    /* Basic factor update algorithms */
    void _als(void);
    void _recurrent_als(void);
};

/* General ALS until the convergence */
class TensorStream_ALS : public TensorStream {
public:
    TensorStream_ALS(DataStream& paperX, const Config& config)
        : TensorStream(paperX, config)
    {
    }
    virtual ~TensorStream_ALS(void) {}

protected:
    virtual void _updateAlgorithm(void) override
    {
        _als();
    }
};

class TensorStream_GD : public TensorStream {
public:
    TensorStream_GD(DataStream& paperX, const Config& config);
    virtual ~TensorStream_GD(void) {}

protected:
    virtual void _updateAlgorithm(void) override;
};

class TensorStream_SGD : public TensorStream {
public:
    TensorStream_SGD(DataStream& paperX, const Config& config);
    virtual ~TensorStream_SGD(void) {}

protected:
    virtual void _updateAlgorithm(void) override;
};
