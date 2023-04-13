#ifndef EMB_H
#define EMB_H

#include <iostream>
// #include <Eigen/Dense>
#include <random>
#include <complex>
#include <mutex>
#include "math.h"
#include <string.h>
#include <stdlib.h>

#include "net.h"

// using namespace Eigen;
using namespace std;

template <typename T>
T Mean(T *vec, int dim) {
    T res = 0;
    for (int d = 0; d < dim; d++) res += vec[d] / dim;
    return res;
}

template <typename T>
void Mul(T *vec1, float alpha, int dim) {
    for (int d = 0; d < dim; d++) vec1[d] *= alpha;
}

template <typename T>
void Add(T *vec1, T *vec2, int dim) {
    for (int d = 0; d < dim; d++) vec1[d] += vec2[d];
}

template <typename T>
void Fill(T *vec, int dim, float value) {
    for (int i = 0; i < dim; i++) vec[i] = value;
}

template <typename T>
inline T DotProd(T *vec1, T *vec2, int dim) {
    T res = 0;
    for (int i = 0; i < dim; i++) res += vec1[i] * vec2[i];
    return res;
}

template <typename T>
void AddAlpha(T *vec1, T *vec2, int dim, float alpha) {
    for (int i = 0; i < dim; i++) vec1[i] += alpha * vec2[i];
}

template <typename T>
void UpdateEmbRiemannian(T *emb, T *grad, int dim, float step_size, int order) {
    T ip;
    float grad_length;

    ip = DotProd(emb, grad, dim);
    switch (order) {
    case 1:
        for (int i = 0; i < dim; i++) emb[i] += step_size * (grad[i] - ip * emb[i]);
        break;
    case 2:
        Mul(grad, step_size, dim);
        grad_length = sqrt(DotProd(grad, grad, dim) );
        // cout << grad_length << endl;
        for (int i = 0; i < dim; i++) emb[i] = cos(grad_length) * emb[i] + sin(grad_length) * grad[i] / (grad_length + 1e-8);
        break;
    default:
        cout << "ERROR: Riemann order " << order << " is invalid" << endl;
        exit(1);
        break;
    }
}

template <typename T>
inline T DotProdOfs(T *emb1, T *emb2, int dim, T ofs) {
    if (ofs < 1e-6) return DotProd(emb1, emb2, dim);
    T res = 0;
    for (int i = 0; i < dim; i++) res += (emb1[i] + ofs) * (emb2[i] + ofs);
    return res;
}

template <typename T>
inline void Normalize(T *emb, int dim, float length = 1.) {
    T m = sqrt(DotProd(emb, emb, dim));
    for (int i = 0; i < dim; i++) emb[i] = emb[i] * length / m;
}

template <typename T>
inline void NormalizeOfs(T *emb, int dim, float length, float ofs) {
    // normalize embedding on the sphere radius=length, orgin points is (ofs, ofs, ...)
    if (ofs < 1e-6) return Normalize(emb, dim, length);
    T m = sqrt(DotProdOfs(emb, emb, dim, -ofs));
    for (int i = 0; i < dim; i++) emb[i] = ofs + (emb[i] - ofs) * length / m;
}

template <typename T>
void NormalizeMat(T *mat, int dim, float length = 1.) {
    for (int i = 0; i < dim; i++) Normalize(mat + i * dim, dim, length);
}

template <typename T>
inline T Sigmoid(T x) {
    return 1 / (1 + exp(-x));
}

template <typename T>
void PrintEmb(T *emb, int dim) {
    for (int d = 0; d < dim; d++) cout << emb[d] << " ";
    cout << endl;
}

template <typename T>
void CopyEmb(T *embs, T *embt, int dim) {
    for (int d = 0; d < dim; d++) embt[d] = embs[d];
}

template <typename T>
inline void ProjSphere(T *emb, T *emb_grad, int dim) {}

template <typename T>
T MapScore(T *emb_src, T *uni, T *emb_tgt, int dim) {
    // return  <emb2|U|emb1>
    T res = 0;
    for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++)
        res += emb_src[j] * uni[i * dim + j] * emb_tgt[i];
    return res;
}

template <typename T>
void MatVecProd(T *mat, T *vec, T *res, int dim, bool mat_transpose = false) {
    // return mat.matmul(vec)
    // if transpose, return mat.T.matmul(vec)
    if (mat_transpose) for (int i = 0; i < dim; i++) {
        res[i] = 0;
        for (int j = 0; j < dim; j++) res[i] += mat[j * dim + i] * vec[j];
    } else for (int i = 0; i < dim; i++) {
        res[i] = 0;
        for (int j = 0; j < dim; j++) res[i] += mat[i * dim + j] * vec[j];
    }
}

long long fast_pow(int x, int m);

class ConventionalEmbedding {
public:
    int dim;
    float *embx;
    float *embh;

    ConventionalEmbedding() {};
    ~ConventionalEmbedding() {};
};

class LayerWeight {
public:
    int L;
    int num_memblock;
    float *weight;
    float *grad;
    float *state_sum;

    LayerWeight() {};
    ~LayerWeight() {
        free(weight);
    }

    void InitLayerWeight(int num_memblock, int L, bool init_grad);
    float GetWeight(int u, int l);
    float *GetWeightPtr(int u);
    void UpdateWeightSGD(int u, float *grad, float step_size);
    void OutputWeight(const char *path);
};

class TUTable{
public:
    int L, C, *R;
    vector<int>row_table;

    void ReadTUConfig(const char *path, int num_memblock);
};

class TensorizedEmbedding {
public:
    int L; // rank
    int C; // order
    int R;

    bool compress_R;
    int *Rs;
    long long *R_ofs;
    long long sum_of_Rs;

    int dim; 
    bool use_h;
    bool init_grad;

    // embedding
    float **embx;
    float **embx_grad;
    float **embh;
    float **embh_grad;

    // batch updating
    mutex **embx_mutex;
    int **embx_batch_counter;
    mutex **embh_mutex;
    int **embh_batch_counter;

    // adagrad & rmsstop
    float **embx_state_sum;
    float **embh_state_sum;

    // rmsstop
    float gamma;

    TensorizedEmbedding() {};
    ~TensorizedEmbedding() {
        delete [] embx;
        if (use_h) delete [] embh;
    };

    void InitTensorizedEmbedding(int L, int C, int R, int dim, float ofs, bool use_h, const char *opt_str);
    void InitTensorizedEmbedding(int L, int C, int *R, int dim, float ofs, bool use_h, const char *opt_str);

    void UpdateEmbSGD(int l, int r, int c, float *grad, int h, int dim, float step_size);
    void UpdateEmbBSGD(int l, int r, int c, float *grad, int h, int dim, float step_size, int batch_size, int riemann_order);
    void UpdateEmbAdagrad(int l, int r, int c, float *grad, int h, int dim, float step_size, int batch_size, bool rmsstop, int riemann_order);

    void OffsetAllEmbedding(float ofs);
    void NormalizeAllEmbedding(float length);
    void OutputTensorEmbedding(const char *path, float ofs, TUTable table);

    float *GetEmbx(int l, int r, int c);
    float *GetEmbh(int l, int r, int c);
    float *GetEmbxGrad(int l, int r, int c);
    float *GetEmbhGrad(int l, int r, int c);
    mutex *GetEmbxMutex(int l, int r, int c);
    mutex *GetEmbhMutex(int l, int r, int c);
    int *GetEmbxBatchCount(int l, int r, int c);
    int *GetEmbhBatchCount(int l, int r, int c);
    float *GetEmbxStateSum(int l, int r, int c);
    float *GetEmbhStateSum(int l, int r, int c);

    void LoadTensorEmbedding(const char *path, TUTable tutab);
};

class Mapping {
public:
    int C;
    int dim;
    int size; // size = dim * dim
    float *weight;

    Mapping() {};
    ~Mapping() {};

    void InitMapping(int C, int dim);
    float *GetMapping(int c);
    void OutputMapping(const char *path);
};

class Trainer {
private:
    bool reg_nce;
    float regw;
    int sphere_norm;
public:
    Trainer() {};
    Trainer(bool reg_nce = true, float regw = 0.01, int sphere_norm = 3) {};
};

// projUNN-T by Yiyang Ling
// void ProjUNN(float *u, float *grad_u, const int d);

#endif