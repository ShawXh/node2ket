#include "emb.h"

// using namespace Eigen;
using namespace std;

long long fast_pow(int x, int m) {
    long long res = 1;
    long long y = x;
    while (m) {
        if (m & 1) res *= y;
        y = y * y;
        m >>= 1;
    }
    return res;
}



void LayerWeight::InitLayerWeight(int num_memblock, int L, bool init_grad) {
    this->num_memblock = num_memblock;
    this->L = L;
    this->weight = (float *)malloc((long long)num_memblock * L * sizeof(float));
    for (long long i = 0; i < num_memblock * L; i++) this->weight[i] = 1./sqrt((float)L);

    if (init_grad) {
        this->grad = (float *)malloc((long long)num_memblock * L * sizeof(float));
        for (long long i = 0; i < num_memblock * L; i++) this->grad[i] = 0;
    }
}

float LayerWeight::GetWeight(int u, int l) {
    return this->weight[(long long)u * this->L + l];
}

float * LayerWeight::GetWeightPtr(int u) {
    return &this->weight[(long long)u * this->L];
}

void LayerWeight::UpdateWeightSGD(int u, float *grad, float step_size) {
    float *w = this->GetWeightPtr(u);
    for (int l = 0; l < this->L; l++) w[l] += step_size * grad[l];
}

void LayerWeight::OutputWeight(const char *path) {
    // TU embeddings
    FILE *fo = fopen(path, "wb");
    fprintf(fo, "%d %d\n", this->num_memblock, this->L);
    for (int u = 0; u < this->num_memblock; u++) {
        for (int l = 0; l < this->L; l++) 
            fprintf(fo, "%f ", this->GetWeight(u, l));
        fprintf(fo, "\n");
    }
    
    fclose(fo);

    printf("Writiing layer weights to %s\n", path);
}


void TensorizedEmbedding::InitTensorizedEmbedding(
    int L, int C, int R, int dim, float ofs, bool use_h, const char *opt_str) {

    this->use_h = use_h;
    this->L = L;
    this->C = C;
    this->R = R;
    this->dim = dim;;
    this->compress_R = false;

    float *vec;

    bool init_grad = strcmp(opt_str, "sgd");
    bool adagrad = !strcmp(opt_str, "adagrad");
    bool rmsprop = !strcmp(opt_str, "rmsprop");

    if (rmsprop) {this->gamma = 0.1; cout << "Initializing optimizer... rmsprop gamma: " << this->gamma << endl;}

    this->embx = new float * [L];
    if (init_grad) {
        this->embx_grad = new float * [L];
        this->embx_mutex = new mutex * [L];
        this->embx_batch_counter = new int * [L];
        if (adagrad || rmsprop) this->embx_state_sum = new float * [L];
    }
    for (int l = 0; l < L; l++) {
        // printf("Initialize layer %d \n", l);
        this->embx[l] = (float *)malloc((long long)C * R * dim * sizeof(float));
        if (init_grad) {
            this->embx_grad[l] = (float *)malloc((long long)C * R * dim * sizeof(float));
            this->embx_mutex[l] = (mutex *)malloc((long long)C * R * sizeof(mutex));
            this->embx_batch_counter[l] = (int *)malloc((long long)C * R * sizeof(int));
        }
        if (adagrad || rmsprop) this->embx_state_sum[l] = (float *)malloc((long long)C * R * sizeof(float));
        // posix_memalign((void **)&this->embx[l], dim, (long long)C * R * dim * sizeof(float));
        if (this->embx[l] == NULL) {printf("Error: memory allocation failed on embx\n"); exit(1);}

        for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) {
            vec = this->GetEmbx(l, r, c);
            for (int d = 0; d < dim; d++) vec[d] = (rand() / (float)RAND_MAX - 0.5) / dim;
            // Normalize(vec, dim, 1.);
            if (init_grad) {
                vec = this->GetEmbxGrad(l, r, c);
                for (int d = 0; d < dim; d++) vec[d] = 0;
                this->embx_batch_counter[l][r * this->C + c] = 0;
                if (adagrad || rmsprop) this->embx_state_sum[l][r * this->C + c] = 0;
            }
        }
    }
    this->OffsetAllEmbedding(ofs);

    if (use_h) {
        this->embh = new float * [L];
        if (init_grad) {
            this->embh_grad = new float * [L];
            this->embh_mutex = new mutex * [L];
            this->embh_batch_counter = new int * [L];
            if (adagrad || rmsprop) this->embh_state_sum = new float * [L];
        }
        for (int l = 0; l < L; l++) {
            this->embh[l] = (float *)malloc((long long)C * R * dim * sizeof(float));
            // posix_memalign((void **)&this->embh[l], dim, (long long)C * R * dim * sizeof(float));
            if (init_grad) {
                this->embh_grad[l] = (float *)malloc((long long)C * R * dim * sizeof(float));
                this->embh_mutex[l] = (mutex *)malloc((long long)C * R * sizeof(mutex));
                this->embh_batch_counter[l] = (int *)malloc((long long)C * R * sizeof(int));
                if (adagrad || rmsprop) this->embh_state_sum[l] = (float *)malloc((long long)C * R * sizeof(float));
            }
            if (this->embh[l] == NULL) { printf("Error: memory allocation failed on embh\n"); exit(1); }

            for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) {
                for (int d = 0; d < dim; d++) 
                    this->embh[l][(long long)(r * C + c) * dim + d] = this->embx[l][(long long)(r * C + c) * dim + d];
                if (init_grad) {
                    vec = this->GetEmbhGrad(l, r, c);
                    for (int d = 0; d < dim; d++) vec[d] = 0;
                    this->embh_batch_counter[l][r * this->C + c] = 0;
                    if (adagrad || rmsprop) this->embh_state_sum[l][r * this->C + c] = 0;
                }
            }
        }
    }

    printf("Finished initializing tensor units\n");
}

void TensorizedEmbedding::InitTensorizedEmbedding(
    int L, int C, int *R, int dim, float ofs, bool use_h, const char *opt_str) {

    this->use_h = use_h;
    this->L = L;
    this->C = C;
    this->Rs = new int [C];
    this->R_ofs = new long long [C];
    this->compress_R = true;
    for (int c = 0; c < C; c++) {
        this->Rs[c] = R[c];
        this->R_ofs[c] = 0;
    }
    for (int c = 1; c < C; c++) {
        this->R_ofs[c] += this->R_ofs[c-1];
        this->R_ofs[c] += this->Rs[c-1];
    }
    this->sum_of_Rs = this->R_ofs[C-1] + this->Rs[C-1];
    cout << "Sum of rows: " << this->sum_of_Rs << endl;
    this->dim = dim;;

    float *vec;

    bool init_grad = strcmp(opt_str, "sgd");
    bool adagrad = !strcmp(opt_str, "adagrad");
    bool rmsprop = !strcmp(opt_str, "rmsprop");

    if (rmsprop) {this->gamma = 0.1; cout << "Initializing optimizer... rmsprop gamma: " << this->gamma << endl;}

    this->embx = new float * [L];
    if (init_grad) {
        this->embx_grad = new float * [L];
        this->embx_mutex = new mutex * [L];
        this->embx_batch_counter = new int * [L];
        if (adagrad || rmsprop) this->embx_state_sum = new float * [L];
    }
    for (int l = 0; l < L; l++) {
        // printf("Initialize layer %d \n", l);
        this->embx[l] = (float *)malloc(this->sum_of_Rs * dim * sizeof(float));
        if (init_grad) {
            this->embx_grad[l] = (float *)malloc(this->sum_of_Rs * dim * sizeof(float));
            this->embx_mutex[l] = (mutex *)malloc(this->sum_of_Rs * sizeof(mutex));
            this->embx_batch_counter[l] = (int *)malloc(this->sum_of_Rs * sizeof(int));
        }
        if (adagrad || rmsprop) this->embx_state_sum[l] = (float *)malloc(this->sum_of_Rs * sizeof(float));
        // posix_memalign((void **)&this->embx[l], dim, this->sum_of_Rs * dim * sizeof(float));
        if (this->embx[l] == NULL) {printf("Error: memory allocation failed on embx\n"); exit(1);}

        for (int c = 0; c < C; c++) for (int r = 0; r < this->Rs[c]; r++) {
            vec = this->GetEmbx(l, r, c);
            for (int d = 0; d < dim; d++) vec[d] = (rand() / (float)RAND_MAX - 0.5) / dim;
            // Normalize(vec, dim, 1.);
            if (init_grad) {
                vec = this->GetEmbxGrad(l, r, c);
                for (int d = 0; d < dim; d++) vec[d] = 0;
                this->embx_batch_counter[l][r + this->R_ofs[c]] = 0;
                if (adagrad || rmsprop) this->embx_state_sum[l][r + this->R_ofs[c]] = 0;
            }
        }
    }
    this->OffsetAllEmbedding(ofs);

    if (use_h) {
        this->embh = new float * [L];
        if (init_grad) {
            this->embh_grad = new float * [L];
            this->embh_mutex = new mutex * [L];
            this->embh_batch_counter = new int * [L];
            if (adagrad || rmsprop) this->embh_state_sum = new float * [L];
        }
        for (int l = 0; l < L; l++) {
            this->embh[l] = (float *)malloc(this->sum_of_Rs * dim * sizeof(float));
            // posix_memalign((void **)&this->embh[l], dim, this->sum_of_Rs * dim * sizeof(float));
            if (init_grad) {
                this->embh_grad[l] = (float *)malloc(this->sum_of_Rs * dim * sizeof(float));
                this->embh_mutex[l] = (mutex *)malloc(this->sum_of_Rs * sizeof(mutex));
                this->embh_batch_counter[l] = (int *)malloc(this->sum_of_Rs * sizeof(int));
                if (adagrad || rmsprop) this->embh_state_sum[l] = (float *)malloc(this->sum_of_Rs * sizeof(float));
            }
            if (this->embh[l] == NULL) { printf("Error: memory allocation failed on embh\n"); exit(1); }

            for (int c = 0; c < C; c++) for (int r = 0; r < this->Rs[c]; r++) {
                for (int d = 0; d < dim; d++) 
                    this->embh[l][(long long)(r + this->R_ofs[c]) * dim + d] = this->embx[l][(long long)(r + this->R_ofs[c]) * dim + d];
                if (init_grad) {
                    vec = this->GetEmbhGrad(l, r, c);
                    for (int d = 0; d < dim; d++) vec[d] = 0;
                    this->embh_batch_counter[l][r + this->R_ofs[c]] = 0;
                    if (adagrad || rmsprop) this->embh_state_sum[l][r + this->R_ofs[c]] = 0;
                }
            }
        }
    }

    printf("Finished initializing tensor units\n");
}

void TensorizedEmbedding::UpdateEmbSGD(int l, int r, int c, float *grad, int h, int dim, float step_size) {
    float *emb;
    if (h) emb = this->GetEmbh(l, r, c);
    else emb = this->GetEmbx(l, r, c);
    for (int i = 0; i < dim; i++) emb[i] += step_size * grad[i];
}

void TensorizedEmbedding::UpdateEmbBSGD(int l, int r, int c, float *grad, int h, int dim, float step_size, int batch_size, int riemann_order) {
    // if (h) lock_guard<mutex> g(*GetEmbhMutex(l, r, c));
    // else lock_guard<mutex> g(*GetEmbxMutex(l, r, c));

    float *emb, *emb_grad;
    int *count;

    if (h) emb_grad = this->GetEmbhGrad(l, r, c);
    else emb_grad = this->GetEmbxGrad(l, r, c);
    Add(emb_grad, grad, dim);
    if (h) count = this->GetEmbhBatchCount(l, r, c);
    else count = this->GetEmbxBatchCount(l, r, c);
    (*count)++;

    if (*count < batch_size) return;
    else {
        if (h) emb = this->GetEmbh(l, r, c);
        else emb = this->GetEmbx(l, r, c);
        if (riemann_order) UpdateEmbRiemannian(emb, emb_grad, dim, step_size / *count, riemann_order);
        else AddAlpha(emb, emb_grad, dim, step_size / *count);
        *count = 0;
        Fill(emb_grad, dim, 0);
    }
}

void TensorizedEmbedding::UpdateEmbAdagrad(int l, int r, int c, float *grad, int h, int dim, float step_size, int batch_size, bool rmsprop, int riemann_order) {
    // if (h) lock_guard<mutex> g(*GetEmbhMutex(l, r, c));
    // else lock_guard<mutex> g(*GetEmbxMutex(l, r, c));

    float *emb, *emb_grad, std_value;
    int *count;

    if (h) emb_grad = this->GetEmbhGrad(l, r, c);
    else emb_grad = this->GetEmbxGrad(l, r, c);
    Add(emb_grad, grad, dim);
    if (h) count = this->GetEmbhBatchCount(l, r, c);
    else count = this->GetEmbxBatchCount(l, r, c);
    (*count)++;

    if (*count < batch_size) return;
    else {
        // average
        Mul(emb_grad, 1./(*count), dim);
        if (h) {
            if (rmsprop) *(this->GetEmbhStateSum(l, r, c)) = (1 - this->gamma) * DotProd(emb_grad, emb_grad, dim) / dim + this->gamma * *(this->GetEmbhStateSum(l, r, c));
            else *(this->GetEmbhStateSum(l, r, c)) += DotProd(emb_grad, emb_grad, dim);
            std_value = sqrt(*(this->GetEmbhStateSum(l, r, c))) + 1e-9;
            emb = this->GetEmbh(l, r, c);
        } else {
            if (rmsprop) *(this->GetEmbxStateSum(l, r, c)) = (1 - this->gamma) * DotProd(emb_grad, emb_grad, dim) / dim + this->gamma * *(this->GetEmbxStateSum(l, r, c));
            else *(this->GetEmbxStateSum(l, r, c)) += DotProd(emb_grad, emb_grad, dim) / dim;
            std_value = sqrt(*(this->GetEmbxStateSum(l, r, c))) + 1e-9;
            emb = this->GetEmbx(l, r, c);
        }

        if (riemann_order) UpdateEmbRiemannian(emb, emb_grad, dim, step_size / std_value, riemann_order);
        else AddAlpha(emb, emb_grad, dim, step_size / std_value);
        
        *count = 0;
        Fill(emb_grad, dim ,0);
    }
}


void TensorizedEmbedding::OffsetAllEmbedding(float ofs) {
    if (ofs < 1e-8) return;

    float *vec;
    
    for (int l = 0; l < this->L; l++) 
    for (int r = 0; r < this->R; r++)
    for (int c = 0; c < this->C; c++) 
    {
        vec = this->GetEmbx(l, r, c);
        for (int d = 0; d < this->dim; d++)
            vec[d] += ofs;
        if (this->use_h) {
            vec = this->GetEmbh(l, r, c);
            for (int d = 0; d < this->dim; d++)
                vec[d] += ofs;
        }
    }

    // printf("Finished offset tensor units with %f\n", ofs);
}

void TensorizedEmbedding::NormalizeAllEmbedding(float length) {
    float *vec;
    
    for (int l = 0; l < this->L; l++) 
    for (int r = 0; r < this->R; r++)
    for (int c = 0; c < this->C; c++) 
    {
        vec = this->GetEmbx(l, r, c);
        Normalize(vec, this->dim, length);
        if (this->use_h) {
            vec = this->GetEmbh(l, r, c);
            Normalize(vec, this->dim, length);
        }
    }

    // printf("Finished offset tensor units with %f\n", ofs);
}

void TensorizedEmbedding::OutputTensorEmbedding(const char *path, float ofs, TUTable tutab) {
    float *vec;
    // TU embeddings
    FILE *fo = fopen(path, "wb");

    if (tutab.row_table.size() == 0) {
        fprintf(fo, "L=%d C=%d R=%d dim=%d\n", this->L, this->C, this->R, this->dim);
        for (int l = 0; l < this->L; l++) 
        for (int c = 0; c < this->C; c++) 
        for (int r = 0; r < this->R; r++) {
            fprintf(fo, "%d %d %d ", l, c, r);
            vec = this->GetEmbx(l, r, c);
            for (int d = 0; d < this->dim; d++)
                fprintf(fo, "%.10f ", vec[d] + ofs);
            fprintf(fo, "\n");
        }
        fclose(fo);
    } else {
        fprintf(fo, "L=%d C=%d R=", this->L, this->C);
        for (int c = 0; c < this->C; c++)
            fprintf(fo, "%d ", tutab.R[c]);
        fprintf(fo, "dim=%d\n", this->dim);
        for (int l = 0; l < this->L; l++) 
        for (int c = 0; c < this->C; c++) 
        for (int r = 0; r < tutab.R[c]; r++) {
            fprintf(fo, "%d %d %d ", l, c, r);
            vec = this->GetEmbx(l, r, c);
            for (int d = 0; d < this->dim; d++)
                fprintf(fo, "%.10f ", vec[d] + ofs);
            fprintf(fo, "\n");
        }
        fclose(fo);
    }

    printf("Writiing TU Embedding to %s\n", path);
}

float *TensorizedEmbedding::GetEmbx(int l, int r, int c) {
    if (!this->compress_R) return &(this->embx[l][(r * this->C + c) * this->dim]);
    else return &(this->embx[l][(r + this->R_ofs[c]) * this->dim]);
}

float *TensorizedEmbedding::GetEmbh(int l, int r, int c) {
    cout << "WARNING: not implemented for getembh" << endl;
    if (!this->compress_R) return &(this->embh[l][(r * this->C + c) * this->dim]);
    else return &(this->embh[l][(r + this->R_ofs[c]) * this->dim]);
}

float *TensorizedEmbedding::GetEmbxGrad(int l, int r, int c) {
    if (!this->compress_R) return &(this->embx_grad[l][(r * this->C + c) * this->dim]);
    else return &(this->embx_grad[l][(r + this->R_ofs[c]) * this->dim]);
}

float *TensorizedEmbedding::GetEmbhGrad(int l, int r, int c) {
    return &(this->embh_grad[l][(r * this->C + c) * this->dim]);
}

mutex *TensorizedEmbedding::GetEmbxMutex(int l, int r, int c) {
    if (!this->compress_R) return &(this->embx_mutex[l][(r * this->C + c)]);
    else return &(this->embx_mutex[l][(r + this->R_ofs[c])]);
}
mutex *TensorizedEmbedding::GetEmbhMutex(int l, int r, int c) {
    return &(this->embh_mutex[l][(r * this->C + c)]);
}

int *TensorizedEmbedding::GetEmbxBatchCount(int l, int r, int c) {
    if (!this->compress_R) return &(this->embx_batch_counter[l][(r * this->C + c)]);
    else return &(this->embx_batch_counter[l][(r + this->R_ofs[c])]);
}

int *TensorizedEmbedding::GetEmbhBatchCount(int l, int r, int c) {
    return &(this->embh_batch_counter[l][(r * this->C + c)]);
}

float *TensorizedEmbedding::GetEmbxStateSum(int l, int r, int c) {
    if (!this->compress_R) return &(this->embx_state_sum[l][(r * this->C + c)]);
    else return &(this->embx_state_sum[l][(r + this->R_ofs[c])]);
}

float *TensorizedEmbedding::GetEmbhStateSum(int l, int r, int c) {
    return &(this->embh_state_sum[l][(r * this->C + c)]);
}

void TensorizedEmbedding::LoadTensorEmbedding(const char *path, TUTable tutab) {
    float *vec;
    FILE *embf = fopen(path, "r");
    int tmp_R;
    if (tutab.row_table.size() == 0) {
        fscanf(embf, "L=%d C=%d R=%d dim=%d\n", &this->L, &this->C, &this->R, &this->dim);
    } else {
        fscanf(embf, "L=%d C=%d R=\n", &this->L, &this->C);
        for (int c = 0; c < this->C; c++) {
            fscanf(embf, "%d", &tmp_R);
            this->R = tmp_R > this->R ? tmp_R : this->R;
        }
        fscanf(embf, " dim=%d\n", &this->dim);
    }
    printf("L=%d C=%d R=%d dim=%d\n", this->L, this->C, this->R, this->dim);

    if (tutab.row_table.size() == 0) 
    // no tutable
        this->InitTensorizedEmbedding(this->L, this->C, this->R, this->dim, 0, 0, "sgd");
    else this->InitTensorizedEmbedding(this->L, this->C, tutab.R, this->dim, 0, 0, "sgd");

    int tmpl, tmpc, tmpr;

    for (int l = 0; l < this->L; l++) 
    for (int c = 0; c < this->C; c++) 
    if (tutab.row_table.size() == 0) for (int r = 0; r < this->R; r++) {
        fscanf(embf, "%d %d %d ", &tmpl, &tmpc, &tmpr);
        vec = this->GetEmbx(tmpl, tmpr, tmpc);
        for (int d = 0; d < this->dim; d++)
            fscanf(embf, "%f ", &vec[d]);
        fprintf(embf, "\n");
    } else for (int r = 0; r < tutab.R[c]; r++) {
        fscanf(embf, "%d %d %d ", &tmpl, &tmpc, &tmpr);
        vec = this->GetEmbx(tmpl, tmpr, tmpc);
        for (int d = 0; d < this->dim; d++)
            fscanf(embf, "%f ", &vec[d]);
        fprintf(embf, "\n");
    } 
    fclose(embf);

    printf("Finish loading tensor embeddings\n");
}




void TUTable::ReadTUConfig(const char *fname, int num_memblock) {
    cout << "Reading TU configs..." << endl;
    FILE *fi = fopen(fname, "rb");
    if (fi == NULL) {
        printf("TU config file doesnt exist.\n");
        exit(1);
    }
    int u, r;
    int simple_config;
    fscanf(fi, "%d %d %d\n", &this->L, &this->C, &simple_config);
    this->R = new int [this->C];
    cout << "TU config: L: " << this->L << " C: " << this->C << " ";
    cout << "#Rows: ";
    for (int c = 0; c < this->C; c++) {
        fscanf(fi, "%d", &this->R[c]);
        cout << this->R[c] << " ";
    }
    cout << endl;
    this->row_table.reserve(num_memblock * this->L * this->C);
    this->row_table.resize(num_memblock * this->L * this->C);
    cout << num_memblock << endl;
    for (long long i = 0; i < num_memblock; i++) {
        fscanf(fi, "%d", &u);
        for (int l = 0; l < this->L; l++) for (int c = 0; c < this->C; c++) {
            fscanf(fi, "%d", &r);
            this->row_table[(u * this->L + l) * this->C + c] = r;
        }
    }
    fclose(fi);
}


void Mapping::InitMapping(int C, int dim) {
    this->C = C;
    this->dim = dim;
    this->size = dim * dim;
    this->weight = new float [C * dim * dim];
    for (int c = 0; c < C; c++)
    for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) {
        if (i == j)
            this->weight[c * dim * dim + i * dim + j] = 1.;
        else 
            this->weight[c * dim * dim + i * dim + j] = 0.;
    }
    // PrintEmb(this->weight, dim * dim);
    printf("Finished mapping matrix initialization.\n");
}

float * Mapping::GetMapping(int c) {
    return &this->weight[c * this->size];
}

void Mapping::OutputMapping(const char *path) {
    printf("Writiing mapping matrix...\n");

    FILE *fo = fopen(path, "wb");
    fprintf(fo, "%d %d\n", this->C, this->dim);
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < this->dim; i++) {
            for (int j = 0; j < this->dim; j++) 
            fprintf(fo, "%f ", this->weight[c * this->dim * this->dim + i * this->dim + j]);
            fprintf(fo, "\n");
        }
        fprintf(fo, "\n");
    }
}

