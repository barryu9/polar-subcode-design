#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

inline double sign_value(double value) {
    if (value > 0.0) {
        return 1.0;
    }
    if (value < 0.0) {
        return -1.0;
    }
    return 0.0;
}

inline double f_min_sum(double left, double right) {
    return sign_value(left) * sign_value(right) *
           std::min(std::abs(left), std::abs(right));
}

inline double &p_at(double *p, int beta, int path, int L) {
    return p[static_cast<std::size_t>(beta) * L + path];
}

inline const double &p_at_const(const double *p, int beta, int path, int L) {
    return p[static_cast<std::size_t>(beta) * L + path];
}

inline std::int8_t &c_at(std::int8_t *c, int beta, int column, int twoL) {
    return c[static_cast<std::size_t>(beta) * twoL + column];
}

inline const std::int8_t &c_at_const(
    const std::int8_t *c, int beta, int column, int twoL
) {
    return c[static_cast<std::size_t>(beta) * twoL + column];
}

inline std::int8_t &u_at(std::int8_t *u, int bit, int path, int L) {
    return u[static_cast<std::size_t>(bit) * L + path];
}

inline const std::int8_t &u_at_const(
    const std::int8_t *u, int bit, int path, int L
) {
    return u[static_cast<std::size_t>(bit) * L + path];
}

void update_c(
    std::int8_t *c,
    int phi,
    int N,
    int L,
    const std::int64_t *lambda_offset,
    const std::int64_t *bit_layer_vec,
    int path
) {
    if ((phi % 2) != 1 || phi == N - 1) {
        return;
    }

    const int twoL = 2 * L;
    const int layer = static_cast<int>(bit_layer_vec[phi]);
    for (int i_layer = 0; i_layer < layer; ++i_layer) {
        const int index_1 = static_cast<int>(lambda_offset[i_layer]);
        const int index_2 = static_cast<int>(lambda_offset[i_layer + 1]);
        for (int beta = index_1; beta < 2 * index_1; ++beta) {
            const int left = c_at(c, beta - 1, 2 * path, twoL);
            const int right = c_at(c, beta - 1, 2 * path + 1, twoL);
            c_at(c, beta + index_1 - 1, 2 * path + 1, twoL) =
                static_cast<std::int8_t>((left + right) % 2);
            c_at(c, beta + index_2 - 1, 2 * path + 1, twoL) =
                c_at(c, beta - 1, 2 * path + 1, twoL);
        }
    }

    const int index_1 = static_cast<int>(lambda_offset[layer]);
    const int index_2 = static_cast<int>(lambda_offset[layer + 1]);
    for (int beta = index_1; beta < 2 * index_1; ++beta) {
        const int left = c_at(c, beta - 1, 2 * path, twoL);
        const int right = c_at(c, beta - 1, 2 * path + 1, twoL);
        c_at(c, beta + index_1 - 1, 2 * path, twoL) =
            static_cast<std::int8_t>((left + right) % 2);
        c_at(c, beta + index_2 - 1, 2 * path, twoL) =
            c_at(c, beta - 1, 2 * path + 1, twoL);
    }
}

void propagate_llr(
    int N,
    int phi,
    int L,
    const std::int64_t *lambda_offset,
    const std::int64_t *llr_layer_vec,
    const double *llr,
    double *p,
    const std::int8_t *c
) {
    int n = 0;
    while ((1 << n) < N) {
        ++n;
    }
    const int layer = static_cast<int>(llr_layer_vec[phi]);
    const int twoL = 2 * L;

    for (int path = 0; path < L; ++path) {
        if (phi == 0) {
            int index_1 = static_cast<int>(lambda_offset[n - 1]);
            for (int beta = 0; beta < index_1; ++beta) {
                p_at(p, beta + index_1 - 1, path, L) = f_min_sum(
                    llr[beta], llr[beta + index_1]
                );
            }
            for (int i_layer = n - 2; i_layer >= 0; --i_layer) {
                index_1 = static_cast<int>(lambda_offset[i_layer]);
                const int index_2 = static_cast<int>(lambda_offset[i_layer + 1]);
                for (int beta = 0; beta < index_1; ++beta) {
                    p_at(p, beta + index_1 - 1, path, L) = f_min_sum(
                        p_at_const(p, beta + index_2 - 1, path, L),
                        p_at_const(p, beta + index_1 + index_2 - 1, path, L)
                    );
                }
            }
        } else if (phi == N / 2) {
            int index_1 = static_cast<int>(lambda_offset[n - 1]);
            for (int beta = 0; beta < index_1; ++beta) {
                const int x_tmp = c_at_const(
                    c, beta + index_1 - 1, 2 * path, twoL
                );
                p_at(p, beta + index_1 - 1, path, L) =
                    (1 - 2 * x_tmp) * llr[beta] + llr[beta + index_1];
            }
            for (int i_layer = n - 2; i_layer >= 0; --i_layer) {
                index_1 = static_cast<int>(lambda_offset[i_layer]);
                const int index_2 = static_cast<int>(lambda_offset[i_layer + 1]);
                for (int beta = 0; beta < index_1; ++beta) {
                    p_at(p, beta + index_1 - 1, path, L) = f_min_sum(
                        p_at_const(p, beta + index_2 - 1, path, L),
                        p_at_const(p, beta + index_1 + index_2 - 1, path, L)
                    );
                }
            }
        } else {
            int index_1 = static_cast<int>(lambda_offset[layer]);
            const int index_2 = static_cast<int>(lambda_offset[layer + 1]);
            for (int beta = 0; beta < index_1; ++beta) {
                const int x_tmp = c_at_const(
                    c, beta + index_1 - 1, 2 * path, twoL
                );
                p_at(p, beta + index_1 - 1, path, L) =
                    (1 - 2 * x_tmp) *
                        p_at_const(p, beta + index_2 - 1, path, L) +
                    p_at_const(p, beta + index_1 + index_2 - 1, path, L);
            }
            for (int i_layer = layer - 1; i_layer >= 0; --i_layer) {
                index_1 = static_cast<int>(lambda_offset[i_layer]);
                const int index_2_inner = static_cast<int>(lambda_offset[i_layer + 1]);
                for (int beta = 0; beta < index_1; ++beta) {
                    p_at(p, beta + index_1 - 1, path, L) = f_min_sum(
                        p_at_const(p, beta + index_2_inner - 1, path, L),
                        p_at_const(
                            p, beta + index_1 + index_2_inner - 1, path, L
                        )
                    );
                }
            }
        }
    }
}

void soft_rank(
    int N,
    int L,
    double sigma,
    const double *pm,
    const std::int8_t *u,
    const npy_bool *active,
    bool &lost,
    double &potential
) {
    int active_count = 0;
    bool has_correct = false;
    int correct_index = 0;
    for (int path = 0; path < L; ++path) {
        if (!active[path]) {
            continue;
        }
        ++active_count;
        int sum = 0;
        for (int bit = 0; bit < N; ++bit) {
            sum += static_cast<int>(u_at_const(u, bit, path, L));
        }
        if (!has_correct && sum == 0) {
            has_correct = true;
            correct_index = path;
        }
    }

    lost = !has_correct;
    if (lost) {
        potential = -1.0;
        return;
    }
    if (active_count <= 1) {
        potential = 1.0;
        return;
    }

    const double correct_pm = pm[correct_index];
    double competitor_pm = std::numeric_limits<double>::infinity();
    for (int path = 0; path < L; ++path) {
        if (active[path] && path != correct_index) {
            competitor_pm = std::min(competitor_pm, pm[path]);
        }
    }
    const double raw_gap = competitor_pm - correct_pm;
    potential = raw_gap / (sigma + std::abs(raw_gap));
}

bool correct_path_survives(
    int N,
    int L,
    const std::int8_t *u,
    const npy_bool *active
) {
    for (int path = 0; path < L; ++path) {
        if (!active[path]) {
            continue;
        }
        bool all_zero = true;
        for (int bit = 0; bit < N; ++bit) {
            if (u_at_const(u, bit, path, L) != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            return true;
        }
    }
    return false;
}

void summarize_path_metrics(
    const std::vector<double> &metrics,
    const std::vector<bool> &valid,
    double &entropy,
    double &candidate_count
) {
    entropy = 0.0;
    candidate_count = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (!valid[index]) {
            continue;
        }
        minimum = std::min(minimum, metrics[index]);
        candidate_count += 1.0;
    }
    if (candidate_count <= 0.0 || !std::isfinite(minimum)) {
        candidate_count = 0.0;
        return;
    }

    std::vector<double> weights(metrics.size(), 0.0);
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (!valid[index]) {
            continue;
        }
        const double weight = std::exp(
            std::max(-745.0, -(metrics[index] - minimum))
        );
        weights[index] = weight;
        weight_sum += weight;
    }
    if (weight_sum <= 0.0 || !std::isfinite(weight_sum)) {
        entropy = 0.0;
        return;
    }
    for (std::size_t index = 0; index < metrics.size(); ++index) {
        if (weights[index] <= 0.0) {
            continue;
        }
        const double probability = weights[index] / weight_sum;
        entropy -= probability * std::log2(probability);
    }
    if (!std::isfinite(entropy)) {
        entropy = 0.0;
    }
}

void decode_row(
    int N,
    int phi,
    int L,
    int expression_width,
    const std::int64_t *lambda_offset,
    const std::int64_t *llr_layer_vec,
    const std::int64_t *bit_layer_vec,
    std::int8_t action,
    const std::int64_t *expression,
    const double *llr,
    double *pm,
    double *p,
    std::int8_t *c,
    std::int8_t *u,
    npy_bool *active,
    double sigma,
    bool calculate_potential,
    bool &lost,
    double &current_potential,
    double &previous_potential,
    bool calculate_prefix_stats,
    double &surviving_entropy,
    double &surviving_count,
    double &pre_pruning_entropy,
    double &pre_pruning_count
) {
    surviving_entropy = 0.0;
    surviving_count = 0.0;
    pre_pruning_entropy = std::numeric_limits<double>::quiet_NaN();
    pre_pruning_count = 0.0;
    if (calculate_potential) {
        soft_rank(N, L, sigma, pm, u, active, lost, previous_potential);
    } else {
        lost = !correct_path_survives(N, L, u, active);
        previous_potential = 0.0;
    }
    propagate_llr(N, phi, L, lambda_offset, llr_layer_vec, llr, p, c);

    if (action == 0) {
        const double max_float = std::numeric_limits<double>::max();
        std::vector<double> pm_pair(static_cast<std::size_t>(2 * L), max_float);
        std::vector<bool> compare(static_cast<std::size_t>(2 * L), false);
        std::vector<bool> active_before(static_cast<std::size_t>(L), false);
        std::vector<bool> active_after(static_cast<std::size_t>(L), false);

        int active_count = 0;
        for (int path = 0; path < L; ++path) {
            active_before[path] = active[path];
            active_after[path] = active[path];
            if (active[path]) {
                ++active_count;
                const double p_value = p_at_const(p, 0, path, L);
                const bool hard = p_value < 0.0;
                pm_pair[path] = pm[path] + std::abs(p_value) * (hard ? 1.0 : 0.0);
                pm_pair[L + path] =
                    pm[path] + std::abs(p_value) * (hard ? 0.0 : 1.0);
            }
        }

        if (calculate_prefix_stats && active_count > 0) {
            std::vector<bool> valid_candidates(
                static_cast<std::size_t>(2 * L), false
            );
            for (int path = 0; path < L; ++path) {
                valid_candidates[path] = active_before[path];
                valid_candidates[L + path] = active_before[path];
            }
            summarize_path_metrics(
                pm_pair,
                valid_candidates,
                pre_pruning_entropy,
                pre_pruning_count
            );
        }

        const int middle = std::max(1, std::min(2 * active_count, L));
        // Select exactly the list capacity, including deterministic metric ties.
        std::vector<int> ordered(static_cast<std::size_t>(2 * L));
        for (int candidate = 0; candidate < 2 * L; ++candidate) {
            ordered[candidate] = candidate;
        }
        std::sort(ordered.begin(), ordered.end(), [&](int a, int b) {
            return pm_pair[a] < pm_pair[b] ||
                (pm_pair[a] == pm_pair[b] && a < b);
        });
        for (int index = 0; index < middle; ++index) {
            compare[ordered[index]] = true;
        }
        if (active_count == 0) {
            std::fill(compare.begin(), compare.end(), false);
        }

        std::vector<int> available;
        for (int path = 0; path < L; ++path) {
            if (!compare[path] && !compare[L + path]) {
                active_after[path] = false;
                available.push_back(path);
            }
        }

        std::vector<int> clone_targets(static_cast<std::size_t>(L), -1);
        for (int path = 0; path < L; ++path) {
            if (!active_before[path]) {
                continue;
            }
            const int state = static_cast<int>(compare[path]) * 2 +
                              static_cast<int>(compare[L + path]);
            if (state == 3 && !available.empty()) {
                clone_targets[path] = available.back();
                available.pop_back();
            }
        }

        for (int path = 0; path < L; ++path) {
            active[path] = active_after[path];
        }

        for (int path = 0; path < L; ++path) {
            if (active_after[path] && !compare[path] && compare[L + path]) {
                pm[path] = pm_pair[L + path];
                u_at(u, phi, path, L) = 1;
                c_at(c, 0, 2 * path + (phi % 2), 2 * L) = 1;
                update_c(c, phi, N, L, lambda_offset, bit_layer_vec, path);
            } else if (active_after[path] && compare[path] && !compare[L + path]) {
                pm[path] = pm_pair[path];
                u_at(u, phi, path, L) = 0;
                c_at(c, 0, 2 * path + (phi % 2), 2 * L) = 0;
                update_c(c, phi, N, L, lambda_offset, bit_layer_vec, path);
            }

            const int destination = clone_targets[path];
            if (destination < 0) {
                continue;
            }
            for (int beta = 0; beta < N - 1; ++beta) {
                p_at(p, beta, destination, L) =
                    p_at_const(p, beta, path, L);
            }
            for (int bit = 0; bit < N; ++bit) {
                u_at(u, bit, destination, L) = u_at_const(u, bit, path, L);
            }
            for (int beta = 0; beta < N - 1; ++beta) {
                c_at(c, beta, 2 * destination, 2 * L) =
                    c_at_const(c, beta, 2 * path, 2 * L);
                c_at(c, beta, 2 * destination + 1, 2 * L) =
                    c_at_const(c, beta, 2 * path + 1, 2 * L);
            }
            active[destination] = true;
            pm[path] = pm_pair[path];
            pm[destination] = pm_pair[L + path];
            u_at(u, phi, path, L) = 0;
            u_at(u, phi, destination, L) = 1;
            c_at(c, 0, 2 * path + (phi % 2), 2 * L) = 0;
            c_at(c, 0, 2 * destination + (phi % 2), 2 * L) = 1;
            update_c(c, phi, N, L, lambda_offset, bit_layer_vec, path);
            update_c(c, phi, N, L, lambda_offset, bit_layer_vec, destination);
        }
    } else {
        const int twoL = 2 * L;
        for (int path = 0; path < L; ++path) {
            int target = 0;
            for (int slot = 0; slot < expression_width; ++slot) {
                const std::int64_t bit = expression[slot];
                if (bit >= 0) {
                    target ^= static_cast<int>(u_at_const(
                        u, static_cast<int>(bit), path, L
                    ));
                }
            }
            if (!active[path]) {
                continue;
            }
            const double p_value = p_at_const(p, 0, path, L);
            const int hard = p_value < 0.0 ? 1 : 0;
            pm[path] += std::abs(p_value) * (hard != target ? 1.0 : 0.0);
            u_at(u, phi, path, L) = static_cast<std::int8_t>(target);
            c_at(c, 0, 2 * path + (phi % 2), twoL) =
                static_cast<std::int8_t>(target);
            update_c(c, phi, N, L, lambda_offset, bit_layer_vec, path);
        }
    }

    if (calculate_prefix_stats) {
        std::vector<double> surviving_metrics(
            static_cast<std::size_t>(L), 0.0
        );
        std::vector<bool> valid_paths(static_cast<std::size_t>(L), false);
        for (int path = 0; path < L; ++path) {
            if (active[path]) {
                surviving_metrics[path] = pm[path];
                valid_paths[path] = true;
            }
        }
        summarize_path_metrics(
            surviving_metrics,
            valid_paths,
            surviving_entropy,
            surviving_count
        );
    }

    if (calculate_potential) {
        soft_rank(N, L, sigma, pm, u, active, lost, current_potential);
    } else {
        lost = !correct_path_survives(N, L, u, active);
        current_potential = 0.0;
    }
}

bool final_path_failed(
    int N,
    int L,
    const double *pm,
    const std::int8_t *u,
    const npy_bool *active,
    double margin
) {
    bool has_correct = false;
    int correct_path = -1;
    double best_pm = std::numeric_limits<double>::infinity();

    for (int path = 0; path < L; ++path) {
        if (!active[path]) {
            continue;
        }
        best_pm = std::min(best_pm, pm[path]);
        bool all_zero = true;
        for (int bit = 0; bit < N; ++bit) {
            if (u_at_const(u, bit, path, L) != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero && !has_correct) {
            has_correct = true;
            correct_path = path;
        }
    }

    if (!has_correct) {
        return true;
    }
    return pm[correct_path] > best_pm + margin;
}

bool check_shape(
    PyArrayObject *array,
    int ndim,
    const std::vector<npy_intp> &expected
) {
    if (PyArray_NDIM(array) != ndim) {
        return false;
    }
    if (static_cast<int>(expected.size()) != ndim) {
        return false;
    }
    for (int axis = 0; axis < ndim; ++axis) {
        if (expected[axis] >= 0 && PyArray_DIM(array, axis) != expected[axis]) {
            return false;
        }
    }
    return true;
}

void resolve_writeback(PyArrayObject *array) {
    if (array != nullptr) {
        PyArray_ResolveWritebackIfCopy(array);
    }
}

PyObject *step_pxb(PyObject *, PyObject *args) {
    int N = 0;
    int phi = 0;
    int L = 0;
    double sigma = 10.0;
    PyObject *lambda_obj = nullptr;
    PyObject *llr_layer_obj = nullptr;
    PyObject *bit_layer_obj = nullptr;
    PyObject *actions_obj = nullptr;
    PyObject *expression_obj = nullptr;
    PyObject *llr_obj = nullptr;
    PyObject *pm_obj = nullptr;
    PyObject *p_obj = nullptr;
    PyObject *c_obj = nullptr;
    PyObject *u_obj = nullptr;
    PyObject *active_obj = nullptr;
    int return_prefix_stats = 0;

    if (!PyArg_ParseTuple(
        args,
        "iOOOiOOOiOOOOOd|p",
        &N,
        &lambda_obj,
        &llr_layer_obj,
        &bit_layer_obj,
        &phi,
        &actions_obj,
        &expression_obj,
        &llr_obj,
        &L,
        &pm_obj,
        &p_obj,
        &c_obj,
        &u_obj,
        &active_obj,
        &sigma,
        &return_prefix_stats
    )) {
        return nullptr;
    }

    if (N <= 0 || L <= 0 || phi < 0 || phi >= N) {
        PyErr_SetString(PyExc_ValueError, "invalid N, L, or phi");
        return nullptr;
    }

    PyArrayObject *lambda_array = nullptr;
    PyArrayObject *llr_layer_array = nullptr;
    PyArrayObject *bit_layer_array = nullptr;
    PyArrayObject *actions_array = nullptr;
    PyArrayObject *expression_array = nullptr;
    PyArrayObject *llr_array = nullptr;
    PyArrayObject *pm_array = nullptr;
    PyArrayObject *p_array = nullptr;
    PyArrayObject *c_array = nullptr;
    PyArrayObject *u_array = nullptr;
    PyArrayObject *active_array = nullptr;
    PyArrayObject *lost_array = nullptr;
    PyArrayObject *current_array = nullptr;
    PyArrayObject *previous_array = nullptr;
    PyArrayObject *prefix_stats_array = nullptr;
    int n = 0;
    npy_intp M = 0;
    npy_intp expression_width = 0;

    lambda_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        lambda_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    llr_layer_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        llr_layer_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    bit_layer_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        bit_layer_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    actions_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        actions_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY
    ));
    expression_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        expression_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    llr_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        llr_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY | NPY_ARRAY_C_CONTIGUOUS
    ));
    pm_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        pm_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY | NPY_ARRAY_C_CONTIGUOUS
    ));
    p_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        p_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY | NPY_ARRAY_C_CONTIGUOUS
    ));
    c_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        c_obj, NPY_BYTE, NPY_ARRAY_INOUT_ARRAY | NPY_ARRAY_C_CONTIGUOUS
    ));
    u_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        u_obj, NPY_BYTE, NPY_ARRAY_INOUT_ARRAY | NPY_ARRAY_C_CONTIGUOUS
    ));
    active_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        active_obj, NPY_BOOL, NPY_ARRAY_INOUT_ARRAY | NPY_ARRAY_C_CONTIGUOUS
    ));

    if (lambda_array == nullptr || llr_layer_array == nullptr ||
        bit_layer_array == nullptr || actions_array == nullptr ||
        expression_array == nullptr || llr_array == nullptr ||
        pm_array == nullptr || p_array == nullptr || c_array == nullptr ||
        u_array == nullptr || active_array == nullptr) {
        goto fail;
    }

    n = static_cast<int>(std::log2(static_cast<double>(N)));
    M = PyArray_DIM(actions_array, 0);
    expression_width = PyArray_DIM(expression_array, 1);
    if (PyArray_NDIM(lambda_array) != 1 || PyArray_DIM(lambda_array, 0) < n + 1 ||
        PyArray_NDIM(llr_layer_array) != 1 || PyArray_DIM(llr_layer_array, 0) != N ||
        PyArray_NDIM(bit_layer_array) != 1 || PyArray_DIM(bit_layer_array, 0) != N ||
        PyArray_NDIM(actions_array) != 1 ||
        PyArray_NDIM(expression_array) != 2 || PyArray_DIM(expression_array, 0) != M ||
        expression_width < 1 ||
        !check_shape(llr_array, 2, {M, N}) ||
        !check_shape(pm_array, 2, {M, L}) ||
        !check_shape(p_array, 3, {M, N - 1, L}) ||
        !check_shape(c_array, 3, {M, N - 1, 2 * L}) ||
        !check_shape(u_array, 3, {M, N, L}) ||
        !check_shape(active_array, 2, {M, L})) {
        PyErr_SetString(PyExc_ValueError, "invalid P*B decoder array shape");
        goto fail;
    }

    {
        const auto *expression_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(expression_array)
        );
        for (npy_intp row = 0; row < M; ++row) {
            const std::int8_t action = reinterpret_cast<const std::int8_t *>(
                PyArray_DATA(actions_array)
            )[row];
            if (action < 0 || action > 4) {
                PyErr_SetString(PyExc_ValueError, "actions must be in range 0..4");
                goto fail;
            }
            for (npy_intp slot = 0; slot < expression_width; ++slot) {
                const std::int64_t bit = expression_data[
                    row * expression_width + slot
                ];
                if (bit >= phi || bit < -1 || bit >= N) {
                    PyErr_SetString(
                        PyExc_ValueError,
                        "dynamic frozen expression must be padded with -1 and causal"
                    );
                    goto fail;
                }
            }
        }
    }

    {
        npy_intp output_dims[1] = {M};
        lost_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, output_dims, NPY_BOOL)
        );
        current_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, output_dims, NPY_DOUBLE)
        );
        previous_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, output_dims, NPY_DOUBLE)
        );
        if (lost_array == nullptr || current_array == nullptr ||
            previous_array == nullptr) {
            goto fail;
        }
        if (return_prefix_stats) {
            npy_intp prefix_dims[2] = {M, 4};
            prefix_stats_array = reinterpret_cast<PyArrayObject *>(
                PyArray_SimpleNew(2, prefix_dims, NPY_DOUBLE)
            );
            if (prefix_stats_array == nullptr) {
                goto fail;
            }
        }
    }

    {
        const auto *lambda_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(lambda_array)
        );
        const auto *llr_layer_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(llr_layer_array)
        );
        const auto *bit_layer_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(bit_layer_array)
        );
        const auto *actions_data = reinterpret_cast<const std::int8_t *>(
            PyArray_DATA(actions_array)
        );
        const auto *expression_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(expression_array)
        );
        auto *llr_data = reinterpret_cast<double *>(PyArray_DATA(llr_array));
        auto *pm_data = reinterpret_cast<double *>(PyArray_DATA(pm_array));
        auto *p_data = reinterpret_cast<double *>(PyArray_DATA(p_array));
        auto *c_data = reinterpret_cast<std::int8_t *>(PyArray_DATA(c_array));
        auto *u_data = reinterpret_cast<std::int8_t *>(PyArray_DATA(u_array));
        auto *active_data = reinterpret_cast<npy_bool *>(PyArray_DATA(active_array));
        auto *lost_data = reinterpret_cast<npy_bool *>(PyArray_DATA(lost_array));
        auto *current_data = reinterpret_cast<double *>(
            PyArray_DATA(current_array)
        );
        auto *previous_data = reinterpret_cast<double *>(
            PyArray_DATA(previous_array)
        );
        auto *prefix_stats_data = return_prefix_stats
            ? reinterpret_cast<double *>(PyArray_DATA(prefix_stats_array))
            : nullptr;

        Py_BEGIN_ALLOW_THREADS
        for (npy_intp row = 0; row < M; ++row) {
            bool lost = false;
            double current_potential = 0.0;
            double previous_potential = 0.0;
            double surviving_entropy = 0.0;
            double surviving_count = 0.0;
            double pre_pruning_entropy = 0.0;
            double pre_pruning_count = 0.0;
            decode_row(
                N,
                phi,
                L,
                static_cast<int>(expression_width),
                lambda_data,
                llr_layer_data,
                bit_layer_data,
                actions_data[row],
                expression_data + row * expression_width,
                llr_data + row * N,
                pm_data + row * L,
                p_data + row * (N - 1) * L,
                c_data + row * (N - 1) * 2 * L,
                u_data + row * N * L,
                active_data + row * L,
                sigma,
                true,
                lost,
                current_potential,
                previous_potential,
                return_prefix_stats != 0,
                surviving_entropy,
                surviving_count,
                pre_pruning_entropy,
                pre_pruning_count
            );
            lost_data[row] = lost;
            current_data[row] = lost ? 0.0 : std::max(-1.0, std::min(1.0, current_potential));
            previous_data[row] = previous_potential;
            if (return_prefix_stats) {
                prefix_stats_data[row * 4] = surviving_entropy;
                prefix_stats_data[row * 4 + 1] = surviving_count;
                prefix_stats_data[row * 4 + 2] = pre_pruning_entropy;
                prefix_stats_data[row * 4 + 3] = pre_pruning_count;
            }
        }
        Py_END_ALLOW_THREADS
    }

    resolve_writeback(llr_array);
    resolve_writeback(pm_array);
    resolve_writeback(p_array);
    resolve_writeback(c_array);
    resolve_writeback(u_array);
    resolve_writeback(active_array);

    Py_DECREF(lambda_array);
    Py_DECREF(llr_layer_array);
    Py_DECREF(bit_layer_array);
    Py_DECREF(actions_array);
    Py_DECREF(expression_array);
    Py_DECREF(llr_array);
    Py_DECREF(pm_array);
    Py_DECREF(p_array);
    Py_DECREF(c_array);
    Py_DECREF(u_array);
    Py_DECREF(active_array);
    if (return_prefix_stats) {
        return Py_BuildValue(
            "NNNN", lost_array, current_array, previous_array, prefix_stats_array
        );
    }
    return Py_BuildValue("NNN", lost_array, current_array, previous_array);

fail:
    resolve_writeback(llr_array);
    resolve_writeback(pm_array);
    resolve_writeback(p_array);
    resolve_writeback(c_array);
    resolve_writeback(u_array);
    resolve_writeback(active_array);
    Py_XDECREF(lambda_array);
    Py_XDECREF(llr_layer_array);
    Py_XDECREF(bit_layer_array);
    Py_XDECREF(actions_array);
    Py_XDECREF(expression_array);
    Py_XDECREF(llr_array);
    Py_XDECREF(pm_array);
    Py_XDECREF(p_array);
    Py_XDECREF(c_array);
    Py_XDECREF(u_array);
    Py_XDECREF(active_array);
    Py_XDECREF(lost_array);
    Py_XDECREF(current_array);
    Py_XDECREF(previous_array);
    Py_XDECREF(prefix_stats_array);
    return nullptr;
}

PyObject *decode_full_pxb(PyObject *, PyObject *args) {
    int N = 0;
    int L = 0;
    int early_stop = 1;
    int return_decoded = 0;
    double terminal_margin = 1e-4;
    PyObject *lambda_obj = nullptr;
    PyObject *llr_layer_obj = nullptr;
    PyObject *bit_layer_obj = nullptr;
    PyObject *actions_obj = nullptr;
    PyObject *expression_obj = nullptr;
    PyObject *llr_obj = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "iOOOOOOipd|p",
        &N,
        &lambda_obj,
        &llr_layer_obj,
        &bit_layer_obj,
        &actions_obj,
        &expression_obj,
        &llr_obj,
        &L,
        &early_stop,
        &terminal_margin,
        &return_decoded
    )) {
        return nullptr;
    }

    if (N <= 1 || L <= 0) {
        PyErr_SetString(PyExc_ValueError, "N must be greater than 1 and L must be positive");
        return nullptr;
    }

    PyArrayObject *lambda_array = nullptr;
    PyArrayObject *llr_layer_array = nullptr;
    PyArrayObject *bit_layer_array = nullptr;
    PyArrayObject *actions_array = nullptr;
    PyArrayObject *expression_array = nullptr;
    PyArrayObject *llr_array = nullptr;
    PyArrayObject *failed_array = nullptr;
    PyArrayObject *first_loss_array = nullptr;
    PyArrayObject *steps_array = nullptr;
    PyArrayObject *decoded_array = nullptr;

    lambda_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        lambda_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    llr_layer_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        llr_layer_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    bit_layer_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        bit_layer_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    actions_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        actions_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY
    ));
    expression_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        expression_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    llr_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        llr_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    ));

    if (lambda_array == nullptr || llr_layer_array == nullptr ||
        bit_layer_array == nullptr || actions_array == nullptr ||
        expression_array == nullptr || llr_array == nullptr) {
        goto full_fail;
    }

    int n = 0;
    while ((1 << n) < N) {
        ++n;
    }
    if ((1 << n) != N) {
        PyErr_SetString(PyExc_ValueError, "N must be a power of two");
        goto full_fail;
    }

    const npy_intp M = PyArray_NDIM(llr_array) == 2
        ? PyArray_DIM(llr_array, 0) : 0;
    const npy_intp expression_width = PyArray_NDIM(expression_array) == 2
        ? PyArray_DIM(expression_array, 1) : 0;
    if (PyArray_NDIM(lambda_array) != 1 || PyArray_DIM(lambda_array, 0) < n + 1 ||
        PyArray_NDIM(llr_layer_array) != 1 || PyArray_DIM(llr_layer_array, 0) != N ||
        PyArray_NDIM(bit_layer_array) != 1 || PyArray_DIM(bit_layer_array, 0) != N ||
        PyArray_NDIM(actions_array) != 1 || PyArray_DIM(actions_array, 0) != N ||
        PyArray_NDIM(expression_array) != 2 || PyArray_DIM(expression_array, 0) != N ||
        expression_width < 1 ||
        !check_shape(llr_array, 2, {M, N})) {
        PyErr_SetString(PyExc_ValueError, "invalid full-path decoder array shape");
        goto full_fail;
    }

    {
        const auto *actions_data = reinterpret_cast<const std::int8_t *>(
            PyArray_DATA(actions_array)
        );
        const auto *expression_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(expression_array)
        );
        for (int phi = 0; phi < N; ++phi) {
            if (actions_data[phi] < 0 || actions_data[phi] > 5) {
                PyErr_SetString(PyExc_ValueError, "actions must be in range 0..5");
                goto full_fail;
            }
            for (npy_intp slot = 0; slot < expression_width; ++slot) {
                const std::int64_t bit = expression_data[
                    static_cast<npy_intp>(phi) * expression_width + slot
                ];
                if (bit >= phi || bit < -1 || bit >= N) {
                    PyErr_SetString(
                        PyExc_ValueError,
                        "dynamic frozen expression must be padded with -1 and causal"
                    );
                    goto full_fail;
                }
            }
        }
    }

    {
        npy_intp output_dims[1] = {M};
        failed_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, output_dims, NPY_BOOL)
        );
        first_loss_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, output_dims, NPY_INT64)
        );
        steps_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, output_dims, NPY_INT64)
        );
        if (failed_array == nullptr || first_loss_array == nullptr ||
            steps_array == nullptr) {
            goto full_fail;
        }
        if (return_decoded) {
            npy_intp decoded_dims[2] = {M, N};
            decoded_array = reinterpret_cast<PyArrayObject *>(
                PyArray_SimpleNew(2, decoded_dims, NPY_BYTE)
            );
            if (decoded_array == nullptr) {
                goto full_fail;
            }
        }
    }

    {
        const auto *lambda_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(lambda_array)
        );
        const auto *llr_layer_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(llr_layer_array)
        );
        const auto *bit_layer_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(bit_layer_array)
        );
        const auto *actions_data = reinterpret_cast<const std::int8_t *>(
            PyArray_DATA(actions_array)
        );
        const auto *expression_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(expression_array)
        );
        const auto *llr_data = reinterpret_cast<const double *>(
            PyArray_DATA(llr_array)
        );
        auto *failed_data = reinterpret_cast<npy_bool *>(
            PyArray_DATA(failed_array)
        );
        auto *first_loss_data = reinterpret_cast<std::int64_t *>(
            PyArray_DATA(first_loss_array)
        );
        auto *steps_data = reinterpret_cast<std::int64_t *>(
            PyArray_DATA(steps_array)
        );
        auto *decoded_data = return_decoded
            ? reinterpret_cast<std::int8_t *>(PyArray_DATA(decoded_array))
            : nullptr;

        Py_BEGIN_ALLOW_THREADS
        for (npy_intp row = 0; row < M; ++row) {
            std::vector<double> pm(static_cast<std::size_t>(L), 0.0);
            std::vector<double> p(
                static_cast<std::size_t>((N - 1) * L), 0.0
            );
            std::vector<std::int8_t> c(
                static_cast<std::size_t>((N - 1) * 2 * L), 0
            );
            std::vector<std::int8_t> u(
                static_cast<std::size_t>(N * L), 0
            );
            std::vector<npy_bool> active(
                static_cast<std::size_t>(L), static_cast<npy_bool>(0)
            );
            active[0] = static_cast<npy_bool>(1);

            bool failed = false;
            std::int64_t first_loss = -1;
            std::int64_t steps = 0;
            for (int phi = 0; phi < N; ++phi) {
                bool lost = false;
                double current_potential = 0.0;
                double previous_potential = 0.0;
                double unused_surviving_entropy = 0.0;
                double unused_surviving_count = 0.0;
                double unused_pre_entropy = 0.0;
                double unused_pre_count = 0.0;
                decode_row(
                    N,
                    phi,
                    L,
                    static_cast<int>(expression_width),
                    lambda_data,
                    llr_layer_data,
                    bit_layer_data,
                    actions_data[phi],
                    expression_data + static_cast<npy_intp>(phi) * expression_width,
                    llr_data + row * N,
                    pm.data(),
                    p.data(),
                    c.data(),
                    u.data(),
                    active.data(),
                    10.0,
                    false,
                    lost,
                    current_potential,
                    previous_potential,
                    false,
                    unused_surviving_entropy,
                    unused_surviving_count,
                    unused_pre_entropy,
                    unused_pre_count
                );
                steps = static_cast<std::int64_t>(phi + 1);
                if (lost) {
                    failed = true;
                    if (first_loss < 0) {
                        first_loss = static_cast<std::int64_t>(phi);
                    }
                    if (early_stop) {
                        break;
                    }
                }
            }

            if (!early_stop || !failed || steps == N) {
                if (steps == N && !failed) {
                    failed = final_path_failed(
                        N, L, pm.data(), u.data(), active.data(), terminal_margin
                    );
                }
            }
            failed_data[row] = failed ? static_cast<npy_bool>(1) : static_cast<npy_bool>(0);
            first_loss_data[row] = first_loss;
            steps_data[row] = steps;
            if (return_decoded) {
                int best_path = -1;
                double best_pm = std::numeric_limits<double>::infinity();
                for (int path = 0; path < L; ++path) {
                    if (active[path] && pm[path] < best_pm) {
                        best_pm = pm[path];
                        best_path = path;
                    }
                }
                if (best_path < 0) {
                    best_path = 0;
                }
                for (int bit = 0; bit < N; ++bit) {
                    decoded_data[row * N + bit] = u_at_const(
                        u.data(), bit, best_path, L
                    );
                }
            }
        }
        Py_END_ALLOW_THREADS
    }

    Py_DECREF(lambda_array);
    Py_DECREF(llr_layer_array);
    Py_DECREF(bit_layer_array);
    Py_DECREF(actions_array);
    Py_DECREF(expression_array);
    Py_DECREF(llr_array);
    if (return_decoded) {
        return Py_BuildValue(
            "NNNN", failed_array, first_loss_array, steps_array, decoded_array
        );
    }
    return Py_BuildValue("NNN", failed_array, first_loss_array, steps_array);

full_fail:
    Py_XDECREF(lambda_array);
    Py_XDECREF(llr_layer_array);
    Py_XDECREF(bit_layer_array);
    Py_XDECREF(actions_array);
    Py_XDECREF(expression_array);
    Py_XDECREF(llr_array);
    Py_XDECREF(failed_array);
    Py_XDECREF(first_loss_array);
    Py_XDECREF(steps_array);
    Py_XDECREF(decoded_array);
    return nullptr;
}

/*
 * Capture every surviving path after a construction prefix.
 *
 * This is intentionally separate from decode_full_pxb: the latter only
 * needs the best terminal path for Monte-Carlo evaluation, while the
 * low-weight-spectrum estimator needs the complete surviving list and the
 * corresponding polar-input prefix for every path.  The decoder state and
 * pruning rule are shared with the production full-path kernel through
 * decode_row(), so this function does not introduce a second SCL algorithm.
 *
 * Return value:
 *   (u_paths[L, prefix_end], pm[L], active[L])
 * where inactive rows in u_paths/pm are unspecified and must be filtered by
 * active.  u_paths is stored row-major for direct NumPy consumption.
 */
PyObject *decode_prefix_list_pxb(PyObject *, PyObject *args) {
    int N = 0;
    int L = 0;
    int prefix_end = 0;
    PyObject *lambda_obj = nullptr;
    PyObject *llr_layer_obj = nullptr;
    PyObject *bit_layer_obj = nullptr;
    PyObject *actions_obj = nullptr;
    PyObject *expression_obj = nullptr;
    PyObject *llr_obj = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "iOOOOOOii",
        &N,
        &lambda_obj,
        &llr_layer_obj,
        &bit_layer_obj,
        &actions_obj,
        &expression_obj,
        &llr_obj,
        &L,
        &prefix_end
    )) {
        return nullptr;
    }

    if (N <= 1 || L <= 0 || prefix_end <= 0 || prefix_end > N) {
        PyErr_SetString(
            PyExc_ValueError,
            "N, L, or prefix_end is invalid"
        );
        return nullptr;
    }

    PyArrayObject *lambda_array = nullptr;
    PyArrayObject *llr_layer_array = nullptr;
    PyArrayObject *bit_layer_array = nullptr;
    PyArrayObject *actions_array = nullptr;
    PyArrayObject *expression_array = nullptr;
    PyArrayObject *llr_array = nullptr;
    PyArrayObject *u_paths_array = nullptr;
    PyArrayObject *pm_paths_array = nullptr;
    PyArrayObject *active_array = nullptr;

    lambda_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        lambda_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    llr_layer_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        llr_layer_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    bit_layer_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        bit_layer_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    actions_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        actions_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY
    ));
    expression_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        expression_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY
    ));
    llr_array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(
        llr_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    ));

    if (lambda_array == nullptr || llr_layer_array == nullptr ||
        bit_layer_array == nullptr || actions_array == nullptr ||
        expression_array == nullptr || llr_array == nullptr) {
        goto prefix_fail;
    }

    int n = 0;
    while ((1 << n) < N) {
        ++n;
    }
    if ((1 << n) != N) {
        PyErr_SetString(PyExc_ValueError, "N must be a power of two");
        goto prefix_fail;
    }

    if (PyArray_NDIM(lambda_array) != 1 || PyArray_DIM(lambda_array, 0) < n + 1 ||
        PyArray_NDIM(llr_layer_array) != 1 || PyArray_DIM(llr_layer_array, 0) != N ||
        PyArray_NDIM(bit_layer_array) != 1 || PyArray_DIM(bit_layer_array, 0) != N ||
        PyArray_NDIM(actions_array) != 1 || PyArray_DIM(actions_array, 0) != N ||
        PyArray_NDIM(expression_array) != 2 || PyArray_DIM(expression_array, 0) != N ||
        PyArray_DIM(expression_array, 1) < 1 ||
        !check_shape(llr_array, 2, {1, N})) {
        PyErr_SetString(PyExc_ValueError, "invalid prefix-list decoder array shape");
        goto prefix_fail;
    }

    const npy_intp expression_width = PyArray_DIM(expression_array, 1);
    const auto *actions_data = reinterpret_cast<const std::int8_t *>(
        PyArray_DATA(actions_array)
    );
    const auto *expression_data = reinterpret_cast<const std::int64_t *>(
        PyArray_DATA(expression_array)
    );
    for (int phi = 0; phi < N; ++phi) {
        if (actions_data[phi] < 0 || actions_data[phi] > 5) {
            PyErr_SetString(PyExc_ValueError, "actions must be in range 0..5");
            goto prefix_fail;
        }
        for (npy_intp slot = 0; slot < expression_width; ++slot) {
            const std::int64_t bit = expression_data[
                static_cast<npy_intp>(phi) * expression_width + slot
            ];
            if (bit >= phi || bit < -1 || bit >= N) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "dynamic frozen expression must be padded with -1 and causal"
                );
                goto prefix_fail;
            }
        }
    }

    {
        npy_intp u_dims[2] = {L, prefix_end};
        npy_intp pm_dims[1] = {L};
        u_paths_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(2, u_dims, NPY_BYTE)
        );
        pm_paths_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, pm_dims, NPY_DOUBLE)
        );
        active_array = reinterpret_cast<PyArrayObject *>(
            PyArray_SimpleNew(1, pm_dims, NPY_BOOL)
        );
        if (u_paths_array == nullptr || pm_paths_array == nullptr ||
            active_array == nullptr) {
            goto prefix_fail;
        }
    }

    {
        const auto *lambda_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(lambda_array)
        );
        const auto *llr_layer_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(llr_layer_array)
        );
        const auto *bit_layer_data = reinterpret_cast<const std::int64_t *>(
            PyArray_DATA(bit_layer_array)
        );
        const auto *llr_data = reinterpret_cast<const double *>(
            PyArray_DATA(llr_array)
        );
        auto *u_paths_data = reinterpret_cast<std::int8_t *>(
            PyArray_DATA(u_paths_array)
        );
        auto *pm_paths_data = reinterpret_cast<double *>(
            PyArray_DATA(pm_paths_array)
        );
        auto *active_paths_data = reinterpret_cast<npy_bool *>(
            PyArray_DATA(active_array)
        );

        std::fill(
            u_paths_data,
            u_paths_data + static_cast<std::size_t>(L) * prefix_end,
            static_cast<std::int8_t>(0)
        );
        std::fill(
            pm_paths_data,
            pm_paths_data + static_cast<std::size_t>(L),
            0.0
        );
        std::fill(
            active_paths_data,
            active_paths_data + static_cast<std::size_t>(L),
            static_cast<npy_bool>(0)
        );

        Py_BEGIN_ALLOW_THREADS
        std::vector<double> pm(static_cast<std::size_t>(L), 0.0);
        std::vector<double> p(
            static_cast<std::size_t>((N - 1) * L), 0.0
        );
        std::vector<std::int8_t> c(
            static_cast<std::size_t>((N - 1) * 2 * L), 0
        );
        std::vector<std::int8_t> u(
            static_cast<std::size_t>(N * L), 0
        );
        std::vector<npy_bool> active(
            static_cast<std::size_t>(L), static_cast<npy_bool>(0)
        );
        active[0] = static_cast<npy_bool>(1);

        for (int phi = 0; phi < prefix_end; ++phi) {
            bool lost = false;
            double current_potential = 0.0;
            double previous_potential = 0.0;
            double unused_surviving_entropy = 0.0;
            double unused_surviving_count = 0.0;
            double unused_pre_entropy = 0.0;
            double unused_pre_count = 0.0;
            decode_row(
                N,
                phi,
                L,
                static_cast<int>(expression_width),
                lambda_data,
                llr_layer_data,
                bit_layer_data,
                actions_data[phi],
                expression_data + static_cast<npy_intp>(phi) * expression_width,
                llr_data,
                pm.data(),
                p.data(),
                c.data(),
                u.data(),
                active.data(),
                10.0,
                false,
                lost,
                current_potential,
                previous_potential,
                false,
                unused_surviving_entropy,
                unused_surviving_count,
                unused_pre_entropy,
                unused_pre_count
            );
        }

        for (int path = 0; path < L; ++path) {
            active_paths_data[path] = active[path];
            pm_paths_data[path] = pm[path];
            if (!active[path]) {
                continue;
            }
            for (int bit = 0; bit < prefix_end; ++bit) {
                u_paths_data[
                    static_cast<std::size_t>(path) * prefix_end + bit
                ] = u_at_const(u.data(), bit, path, L);
            }
        }
        Py_END_ALLOW_THREADS
    }

    Py_DECREF(lambda_array);
    Py_DECREF(llr_layer_array);
    Py_DECREF(bit_layer_array);
    Py_DECREF(actions_array);
    Py_DECREF(expression_array);
    Py_DECREF(llr_array);
    return Py_BuildValue("NNN", u_paths_array, pm_paths_array, active_array);

prefix_fail:
    Py_XDECREF(lambda_array);
    Py_XDECREF(llr_layer_array);
    Py_XDECREF(bit_layer_array);
    Py_XDECREF(actions_array);
    Py_XDECREF(expression_array);
    Py_XDECREF(llr_array);
    Py_XDECREF(u_paths_array);
    Py_XDECREF(pm_paths_array);
    Py_XDECREF(active_array);
    return nullptr;
}

PyMethodDef module_methods[] = {
    {
        "step_pxb",
        reinterpret_cast<PyCFunction>(step_pxb),
        METH_VARARGS,
        "Run one unified CPU P*B SCL decoder step."
    },
    {
        "decode_full_pxb",
        reinterpret_cast<PyCFunction>(decode_full_pxb),
        METH_VARARGS,
        "Decode complete fixed action paths for a batch of AWGN frames."
    },
    {
        "decode_prefix_list_pxb",
        reinterpret_cast<PyCFunction>(decode_prefix_list_pxb),
        METH_VARARGS,
        "Capture all surviving SCL paths after one construction prefix."
    },
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_scl_step_decoder_cpp",
    "CPU C++ implementation of the unified P*B SCL step kernel.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__scl_step_decoder_cpp(void) {
    import_array();
    return PyModule_Create(&module_definition);
}
