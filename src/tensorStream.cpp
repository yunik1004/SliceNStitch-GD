#include "tensorStream.hpp"
#include "constants.hpp"
#include "loader.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

TensorStream* generateTensorStream(DataStream& paperX, const Config& config)
{
    TensorStream* ts;
    if (config.algo() == "GD") {
        ts = new TensorStream_GD(paperX, config);
    } else if (config.algo() == "SGD") {
        ts = new TensorStream_SGD(paperX, config);
    } else if (config.algo() == "Momentum") {
        ts = new TensorStream_Momentum(paperX, config);
    } else if (config.algo() == "RMSProp") {
        ts = new TensorStream_RMSProp(paperX, config);
    } else {
        ts = new TensorStream(paperX, config);
    }
    return ts;
}

TensorStream::TensorStream(DataStream& paperX, const Config& config)
{
    _paperX = &paperX;
    _config = &config;

    const int numMode = _config->numMode();
    const int unitNum = _config->unitNum();
    const int unitSize = _config->unitSize();
    const int rank = _config->rank();

    std::vector<int> dimension = _config->nonTempDim(); // Dimension of the tensor X
    dimension.push_back(unitNum);

    // Set the computing order
    {
        _compute_order.reserve(numMode);
        _compute_order.push_back(numMode - 1);
        for (int m = 0; m < numMode - 1; ++m) {
            _compute_order.push_back(m);
        }
    }

    _elapsed_time = std::chrono::nanoseconds::zero();

    /* Initialize X */
    {
        _X = new SpTensor_Hash(dimension);

        std::vector<int> coord(numMode);

        while (true) {
            auto e = _paperX->pop();
            if (e == nullptr) {
                break;
            }

            for (int m = 0; m < numMode - 1; ++m) {
                coord[m] = e->nonTempCoord[m];
            }
            coord[numMode - 1] = e->newUnitIdx;
            _X->insert(coord, e->val);
        }
    }

    /* Initialize dX */
    _dX = new SpTensor_dX(dimension);

    /* Initialize lambda, A, AtA */
    {
        _A.resize(numMode);
        _AtA.resize(numMode);

        _als();
        _unnormalize_A(); // Unnormalization
    }
}

TensorStream::~TensorStream(void)
{
    delete _dX;
    delete _X;
}

void TensorStream::updateTensor(const DataStream::Event& e)
{
    const int unitNum = _config->unitNum();
    const int unitSize = _config->unitSize();
    const int numMode = _config->numMode();

    // Clear dX
    _dX->clear();

    // Eventwise
    std::vector<int> coord(numMode);
    for (int m = 0; m < numMode - 1; ++m) {
        coord[m] = e.nonTempCoord[m];
    }
    coord[numMode - 1] = e.newUnitIdx;

    if (e.newUnitIdx != -1) {
        _dX->insert(coord, e.val);
        _X->insert(coord, e.val);
    }

    if (e.newUnitIdx != unitNum - 1) {
        coord[numMode - 1] += 1;
        _dX->insert(coord, -e.val);
        _X->insert(coord, -e.val);
    }
}

void TensorStream::saveFactor(std::string fileName) const
{
    const int numMode = _config->numMode();
    // Save factor matrix
    for (int m = 0; m < numMode; ++m) {
        std::ofstream outFile("factor_matrix/mode_" + std::to_string(m) + "_" + fileName);
        outFile << std::setprecision(std::numeric_limits<double>::digits10 + 2) << _A[m];
        outFile.close();
    }
}

void TensorStream::updateFactor(void)
{
    int _updatePeriod = _config->updatePeriod();
    if (_updatePeriod == 1) {
        const unsigned long long numDelta = _dX->numNnz();
        assert(numDelta >= 0);
        if (numDelta == 0)
            return;
    }

    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    _updateAlgorithm();
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    _elapsed_time += end - begin;

    /* Normalization
    const int numMode = _config->numMode();
    const int rank = _config->rank();

    std::vector<Eigen::ArrayXd> lambdas(numMode);

    // Update lambda (column-wise L_infinity-norm)
    for (int m = 0; m < numMode; ++m) {
        lambdas[m] = _A[m].cwiseAbs().colwise().maxCoeff().array();
        for (int r = 0; r < rank; ++r) { // If lambda is 0, then set to 1
            if (abs(lambdas[m][r]) < TENSOR_MACHINE_EPSILON) {
                lambdas[m][r] = 1;
            }
        }

        // Normalize A
        _A[m] = (_A[m].array().rowwise() / lambdas[m].transpose()).matrix();
    }

    for (int r = 0; r < rank; ++r) {
        _lambda[r] = 1.0;
        for (int m = 0; m < numMode; ++m) {
            _lambda[r] *= lambdas[m][r];
        }
    }

    // Unnormalize A
    _unnormalize_A(); */
}

void TensorStream::updateAtA(void)
{
    if (!_use_AtA) {
        const int numMode = _config->numMode();

        for (int m = 0; m < numMode; m++) {
            _AtA[m] = (_A[m].transpose() * _A[m]).array();
        }
    }
}

double TensorStream::elapsedTime(void) const
{
    return std::chrono::duration<double>(_elapsed_time).count();
}

double TensorStream::density() const
{
    const std::vector<int>& dimension = _X->dimension();
    double _density = (double)(_X->numNnz());
    const int& _numMode = _config->numMode();

    for (int m = 0; m < _numMode; m++)
        _density /= (double)dimension[m];

    return _density;
}

double TensorStream::find_reconst(const std::vector<int>& coord) const
{
    const int rank = _config->rank();
    const int numMode = _config->numMode();

    Eigen::ArrayXd prodMatrix = Eigen::ArrayXd::Ones(rank);
    for (int m = 0; m < numMode; ++m) {
        prodMatrix *= _A[m].row(coord[m]).array();
    }
    prodMatrix *= _lambda;

    return prodMatrix.sum();
}

double TensorStream::rmse(void) const
{
    const std::vector<int>& dimension = _X->dimension();
    const double normX = _X->norm_frobenius();
    const double normX_hat = _norm_frobenius_reconst();
    const double innerprod = _innerprod_X_X_reconst();
    const double normresidual_square = abs(pow(normX, 2) + pow(normX_hat, 2) - 2 * innerprod);

    double error_square = normresidual_square;
    for (int const& modeSize : dimension) {
        error_square /= modeSize;
    }
    return sqrt(error_square);
}

double TensorStream::fitness(void) const
{
    const double normX = _X->norm_frobenius();
    const double normX_hat = _norm_frobenius_reconst();
    const double innerprod = _innerprod_X_X_reconst();
    const double normresidual = sqrt(abs(pow(normX, 2) + pow(normX_hat, 2) - 2 * innerprod));

    return 1 - normresidual / normX;
}

double TensorStream::fitness_latest(void) const
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();

    const double normX_latest = _X->norm_frobenius_latest();

    // Compute the latest normX_hat
    double normX_hat_latest_square = 0;
    {
        const Eigen::MatrixXd lambda_mat = _lambda.matrix();

        Eigen::ArrayXXd coefMatrix = (lambda_mat * lambda_mat.transpose()).array();
        for (int m = 0; m < numMode - 1; ++m) {
            coefMatrix *= _AtA[m];
        }
        {
            const auto A_temporal = _A[numMode - 1].row(dimension[numMode - 1] - 1);
            coefMatrix *= (A_temporal.transpose() * A_temporal).array();
        }

        normX_hat_latest_square = abs(coefMatrix.sum());
    }

    // Compute the latest inner product
    double innerprod_latest = 0;
    {
        const SpTensor_Hash::coord_map& coord_map = _X->elems()[numMode - 1][dimension[numMode - 1] - 1];
        for (auto const& it : coord_map) {
            const std::vector<int> coord_vec = it.first;
            const double& val_real = it.second;
            const double val_reconst = find_reconst(coord_vec);
            innerprod_latest += val_real * val_reconst;
        }
    }

    const double normresidual = sqrt(abs(pow(normX_latest, 2) + normX_hat_latest_square - 2 * innerprod_latest));

    return 1 - normresidual / normX_latest;
}

double TensorStream::error(const std::vector<int>& coord) const
{
    const double val = _X->find(coord);
    const double val_reconst = find_reconst(coord);

    //return abs((val - val_reconst) / val);
    return abs(val - val_reconst);
}

double TensorStream::_norm_frobenius_reconst(void) const
{
    const int numMode = _config->numMode();
    const Eigen::MatrixXd lambda_mat = _lambda.matrix();

    Eigen::ArrayXXd coefMatrix = (lambda_mat * lambda_mat.transpose()).array();
    for (int m = 0; m < numMode; ++m) {
        coefMatrix *= _AtA[m];
    }
    return sqrt(abs(coefMatrix.sum()));
}

double TensorStream::_innerprod_X_X_reconst(void) const
{
    double innerprod = 0;

    const std::vector<SpTensor_Hash::coord_map>& elems = _X->elems()[0];
    for (auto const& coord_map : elems) {
        for (auto const& it : coord_map) {
            const std::vector<int> coord_vec = it.first;
            const double& val_real = it.second;
            const double val_reconst = find_reconst(coord_vec);
            innerprod += val_real * val_reconst;
        }
    }

    return innerprod;
}

inline void TensorStream::_rand_init_A(void)
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();
    const int unitNum = _config->unitNum();
    const int rank = _config->rank();

    std::mt19937* rng = RNG::Instance()->rng();
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int m = 0; m < numMode - 1; ++m) {
        _A[m] = Eigen::MatrixXd::NullaryExpr(dimension[m], rank, [&]() { return dis(*rng); });
    }
    _A[numMode - 1] = Eigen::MatrixXd::NullaryExpr(unitNum, rank, [&]() { return dis(*rng); });

    for (int m = 0; m < numMode; ++m) {
        _AtA[m] = (_A[m].transpose() * _A[m]).array();
    }

    _lambda = Eigen::ArrayXd::Ones(rank);
}

inline void TensorStream::_als_base(void)
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<SpTensor_Hash::row_vector>& elems = _X->elems();

    for (int const& m : _compute_order) {
        const int numIdx = dimension[m];

        // Initialize V
        Eigen::MatrixXd V;
        {
            Eigen::ArrayXXd V_arr = Eigen::ArrayXXd::Ones(rank, rank);
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                V_arr *= _AtA[n];
            }
            V = V_arr.matrix();
        }

        // Compute mttkrp
        Eigen::MatrixXd mkp = Eigen::MatrixXd::Zero(numIdx, rank);
        {
            int idx = 0;
            for (auto const& coord_map : elems[m]) {
                for (auto const& it : coord_map) {
                    const std::vector<int> coord_vec = it.first;
                    const double val = it.second;

                    Eigen::ArrayXd mkp_row = Eigen::ArrayXd::Constant(rank, val);
                    for (int n = 0; n < numMode; ++n) {
                        if (n == m) {
                            continue;
                        }
                        mkp_row *= _A[n].row(coord_vec[n]).array();
                    }
                    mkp.row(idx) += mkp_row.transpose().matrix();
                }
                idx++;
            }
        }

        // Solve mkp / V using householder QR
        _A[m] = V.transpose().householderQr().solve(mkp.transpose()).transpose();

        // Update lambda (column-wise L_infinity-norm)
        _lambda = _A[m].cwiseAbs().colwise().maxCoeff().array();
        for (int r = 0; r < rank; ++r) { // If lambda is 0, then set to 1
            if (abs(_lambda[r]) < TENSOR_MACHINE_EPSILON) {
                _lambda[r] = 1;
            }
        }

        // Normalize A
        _A[m] = (_A[m].array().rowwise() / _lambda.transpose()).matrix();

        // Update AtA
        _AtA[m] = (_A[m].transpose() * _A[m]).array();
    }
}

inline void TensorStream::_unnormalize_A(void)
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();

    // Multiply Nth root of lambda into each factor matrix
    const Eigen::ArrayXd nrootLambda = _lambda.pow(1.0 / numMode);

    _lambda = Eigen::ArrayXd::Ones(rank);
    for (int m = 0; m < numMode; ++m) {
        _A[m] = (_A[m].array().rowwise() * nrootLambda.transpose()).matrix();
        _AtA[m] = (_A[m].transpose() * _A[m]).array();
    }
}

void TensorStream::_als(void)
{
    _rand_init_A(); // Randomly initialize A
    _recurrent_als();
}

void TensorStream::_recurrent_als(void)
{
    // Run ALS until fit change is lower than tolerance
    double fitold = 0;
    for (int i = 0; i < ALS_MAX_ITERS; ++i) {
        _als_base();

        const double fitnew = fitness();
        if (i > 0 && abs(fitold - fitnew) < ALS_FIT_CHANGE_TOL) {
            break;
        }
        fitold = fitnew;
    }
}

void TensorStream_GD::_updateAlgorithm(void)
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();
    const int rank = _config->rank();
    const std::vector<SpTensor_Hash::row_vector>& elems = _X->elems();

    std::vector<Eigen::MatrixXd> gradA(numMode);

    for (int const& m : _compute_order) {
        const int numIdx = dimension[m];

        // Initialize V
        Eigen::MatrixXd V;
        {
            Eigen::ArrayXXd V_arr = Eigen::ArrayXXd::Ones(rank, rank);
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                V_arr *= _AtA[n];
            }
            V = V_arr.matrix();
        }

        // Compute mttkrp
        Eigen::MatrixXd mkp = Eigen::MatrixXd::Zero(numIdx, rank);
        {
            int idx = 0;
            for (auto const& coord_map : elems[m]) {
                for (auto const& it : coord_map) {
                    const std::vector<int> coord_vec = it.first;
                    const double val = it.second;

                    Eigen::ArrayXd mkp_row = Eigen::ArrayXd::Constant(rank, val);
                    for (int n = 0; n < numMode; ++n) {
                        if (n == m) {
                            continue;
                        }
                        mkp_row *= _A[n].row(coord_vec[n]).array();
                    }
                    mkp.row(idx) += mkp_row.transpose().matrix();
                }
                idx++;
            }
        }

        // Compute gradient A
        gradA[m] = _A[m] * V - mkp;
    }

    // Update A and AtA
    for (int m = 0; m < numMode; ++m) {
        _A[m] -= _lr * gradA[m];
        _AtA[m] = (_A[m].transpose() * _A[m]).array();
    }
}

TensorStream_SGD::TensorStream_SGD(DataStream& paperX, const Config& config)
    : TensorStream_GD(paperX, config)
    , _numSample(_config->findAlgoSettings<int>("numSample"))
{
    _use_AtA = false;
}

void TensorStream_SGD::_updateAlgorithm(void)
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();

    std::unordered_set<std::vector<int>> sampledIdx;
    const int numSampleReal = _sampleEntry(sampledIdx);

    // Set gradients
    std::vector<Eigen::MatrixXd> gradAs(numSampleReal);
    {
        int i = 0;
        for (const auto& e : sampledIdx) {
            const double val_real = _X->find(e);
            const double val_reconst = find_reconst(e);

            gradAs[i] = Eigen::MatrixXd::Constant(numMode, rank, (val_reconst - val_real) / numSampleReal);
            for (int m = 0; m < numMode; ++m) {
                for (int n = 0; n < numMode; ++n) {
                    if (n == m) {
                        continue;
                    }
                    gradAs[i].row(m) = gradAs[i].row(m).cwiseProduct(_A[n].row(e[n]));
                }
            }

            ++i;
        }
    }

    // Update factor matrices
    {
        int i = 0;
        for (const auto& e : sampledIdx) {
            for (int m = 0; m < numMode; ++m) {
                _A[m].row(e[m]) -= _lr * gradAs[i].row(m);
            }
            ++i;
        }
    }
}

int TensorStream_SGD::_sampleEntry(std::unordered_set<std::vector<int>>& sampledIdx) const
{
    const std::vector<int>& dimension = _X->dimension();
    const std::vector<std::vector<int>>& nnzIdxLists = _dX->idxLists();
    const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();

    // Sample indices with replacement
    int numdX = 0;
    // Insert changed elements
    for (int const& i : nnzIdxLists[0]) {
        const SpTensor_Hash::coord_map& cmap = elemsdX[0][i];
        for (const auto& it : cmap) {
            sampledIdx.insert(it.first);
            ++numdX;
        }
    }

    // Insert sampled elements
    pickIdx_replacement(dimension, _numSample, sampledIdx);

    // Return the number of sampled entries
    return numdX + _numSample;
}

void TensorStream_SGD::_compute_gradA(std::vector<std::unordered_map<int, Eigen::MatrixXd>>& gradA) const
{
    const int numMode = _config->numMode();
    const int rank = _config->rank();

    std::unordered_set<std::vector<int>> sampledIdx;
    const int numSampleReal = _sampleEntry(sampledIdx);

    // Compute the gradients
    gradA.resize(numMode);
    for (const auto& e : sampledIdx) {
        const double val_real = _X->find(e);
        const double val_reconst = find_reconst(e);

        for (int m = 0; m < numMode; ++m) {
            Eigen::MatrixXd grad = Eigen::MatrixXd::Constant(1, rank, (val_reconst - val_real) / numSampleReal);
            for (int n = 0; n < numMode; ++n) {
                if (n == m) {
                    continue;
                }
                grad = grad.cwiseProduct(_A[n].row(e[n]));
            }

            // Update gradA
            {
                const std::unordered_map<int, Eigen::MatrixXd>::const_iterator& it = gradA[m].find(e[m]);
                const Eigen::MatrixXd preValue = (it != gradA[m].end()) ? it->second : Eigen::MatrixXd::Zero(1, rank);
                gradA[m][e[m]] = preValue + grad;
            }
        }
    }
}

TensorStream_Momentum::TensorStream_Momentum(DataStream& paperX, const Config& config)
    : TensorStream_SGD(paperX, config)
    , _momentum(_config->findAlgoSettings<double>("momentum"))
    , _momentumNew(_config->findAlgoSettings<double>("momentumNew"))
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();
    const int rank = _config->rank();

    /* Initialize V by 0 */
    {
        _V.resize(numMode);

        for (int m = 0; m < numMode; ++m) {
            _V[m] = Eigen::MatrixXd::Zero(dimension[m], rank);
        }
    }
}

void TensorStream_Momentum::_updateAlgorithm(void)
{
    const int numMode = _config->numMode();
    std::vector<std::unordered_map<int, Eigen::MatrixXd>> gradA;
    _compute_gradA(gradA);

    // Downgrades the rows of V which are correspond to dX
    {
        const std::vector<int>& dimension = _X->dimension();
        const std::vector<std::vector<int>>& nnzIdxLists = _dX->idxLists();
        const std::vector<SpTensor_Hash::row_vector>& elemsdX = _dX->elems();

        for (int const& i : nnzIdxLists[0]) {
            const SpTensor_Hash::coord_map& cmap = elemsdX[0][i];
            for (const auto& it : cmap) {
                const std::vector<int>& idx = it.first;
                for (int m = 0; m < numMode; ++m) {
                    _V[m].row(idx[m]) *= _momentumNew;
                }
            }
        }
    }

    // Update factor matrices
    for (int m = 0; m < numMode; ++m) {
        for (const auto& g : gradA[m]) {
            const int mdx = g.first;
            const auto newV = _momentum * _V[m].row(mdx) - _lr * g.second;
            _V[m].row(mdx) = newV;
            _A[m].row(mdx) += newV;
        }
    }
}

TensorStream_RMSProp::TensorStream_RMSProp(DataStream& paperX, const Config& config)
    : TensorStream_SGD(paperX, config)
    , _decay(_config->findAlgoSettings<double>("decay"))
{
    const std::vector<int>& dimension = _X->dimension();
    const int numMode = _config->numMode();

    /* Initialize G by 0 */
    {
        _G.resize(numMode);

        for (int m = 0; m < numMode; ++m) {
            _G[m] = Eigen::VectorXd::Zero(dimension[m]);
        }
    }
}

void TensorStream_RMSProp::_updateAlgorithm(void)
{
    const int numMode = _config->numMode();
    std::vector<std::unordered_map<int, Eigen::MatrixXd>> gradA;
    _compute_gradA(gradA);

    // Update G
    for (int m = 0; m < numMode; ++m) {
        for (const auto& g : gradA[m]) {
            const int mdx = g.first;
            const auto& grad = g.second;
            const double newG = _decay * _G[m][mdx] + (1 - _decay) * grad.squaredNorm();
            _G[m][mdx] = newG;
            _A[m].row(mdx) -= _lr / sqrt(newG + RMSPROP_EPSILON) * grad;
        }
    }
}
