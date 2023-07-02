#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <vector>
#include <random>
#include <mutex>
// #include <utility>
#include <algorithm>
// #include <queue>
// #include <cstring>

#include <net.h>
#include <emb.h>

#define MAX_STRING 100
#define sigmoid_table_size 1024
#define SIGMOID_BOUND 6

using namespace std;

extern std::mt19937 rng;
extern std::uniform_real_distribution<double> randf;

char network_file[MAX_STRING] = "None"; 
char sequence_file[MAX_STRING] = "None"; 
char sub_emb_config_file[MAX_STRING] = "None"; 
char sub_embedding_file_src[MAX_STRING] = "sub_embedding.txt";
char node_embedding_file[MAX_STRING] = "node_embedding.txt";
char layer_weight_file[MAX_STRING] = "layer_weight.txt";
char log_file[MAX_STRING] = "tmpfile/log.txt";

int debug = 0;
int seed = 0;
int log_loss = 0;
int print_progress = 1;

// Basic Parameters

int num_threads = 8;
int arch = 0; // 0 for undirected, 1 for directed.
int outputemb = 0;
int eval_nr = 0;
double regw = 0.1; 
int window_size = 1;
long long total_samples = 300, current_sample_count = 0;
long long num_edges = 0;
float init_rho = 0.1, rho;
int nr_sample_rate = 10;
float margin = 0.1;
float ppralpha = 0.85;
int rw = 0;
int rwr = 0;
int opt = 3; // opt = 1 for sgd, 2 for bsgd, 3 for adagrad
char opt_str[MAX_STRING] = "adagrad";
int obj = 0; // obj = 0 for sigmoid , 1 for marginal triplets
float obj_mix_thresh = 0.3;
char obj_str[MAX_STRING];
int num_neg = 5; // number of negative sampling
int batch_size = 8;
float temp = 1;
int norm_sphere = 1;
int constrain_zero = 1;
int riemann_order = 1;

Network net;
NodeSequences seq;
int net_flag, seq_flag;
TensorizedEmbedding embedding;
LayerWeight layer_weight;
int use_lw = 0;
bool readtu_flag = false;
TUTable tutab;


// params for embedding
int L = 1, C = 0, dim = 0;
int *R;

float *sigmoid_table;

void InitSigmoidTable()
{
    float x;
    sigmoid_table = (float *)malloc((sigmoid_table_size) * sizeof(float));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

float FastSigmoid(float x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

inline int GetRowIdx(int u, int l, int c) {
    // cout << "Getting row idx" << endl;
    if (readtu_flag) return tutab.row_table[(u * L + l) * C + c];
    else return u;
}

inline float * GetEmbVertex(int u, int l, int c, int fetch_h) {
    u = GetRowIdx(u, l, c);
    // cout << "row idx " << u << endl;
    if (fetch_h) return embedding.GetEmbh(l, u, c);
    else return embedding.GetEmbx(l, u, c);
}

void GetFullEmb(int u, float *full_emb) {
    int full_dim = (int)fast_pow(dim, C);
    float *emb1, *emb2, *result_emb;
    int cur_size = dim * dim;
    float lw = 0.;
    // flush
    for (int d = 0; d < full_dim; d++) full_emb[d] = 0.;
    for (int l = 0; l < L; l++) {
        emb1 = GetEmbVertex(u, l, 0, 0);
        if (C == 1) result_emb = emb1;
        else for (int c = 1; c < C; c++) {
            emb2 = GetEmbVertex(u, l, c, 0);
            result_emb = new float [cur_size];
            for (int i = 0; i < cur_size / dim; i++) for (int j = 0; j < dim; j++)
                result_emb[i * dim + j] = emb1[i] * emb2[j];
            if (c > 1) delete [] emb1;
            emb1 = result_emb;
            cur_size *= dim;
        }
        // fill in
        if (use_lw) lw = layer_weight.GetWeight(u, l);
        else lw = 1./sqrt(L);
        for (int d = 0; d < full_dim; d++) {
            full_emb[d] += result_emb[d] * lw * lw;
        }
        if (C != 1) delete [] result_emb;
    }
}

// Regularizer
inline void BackwardReg(float *vec_u, float *vec_v, float *vec_error_u, float *vec_error_v, float weight) {
    if (weight < 1e-6) return;

	float lam_dis = 0, dis2 = 0, g;
	
    // Wasserstein-3
    for (int c = 0; c != dim; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);

    if (dis2 < 1e-8) return;

    lam_dis = 3 * sqrt(dis2) * weight;
    // lam_dis = weight;

    for (int c = 0; c < dim; c++) {
	    g = lam_dis * (vec_u[c] - vec_v[c]);
	    vec_error_u[c] += -g;
	    vec_error_v[c] += g;
	}
}

// inline void BackwardReg(float *vec_u, float *vec_v, float *vec_error_u, float *vec_error_v, float tmp_chipped_layer_ip, float weight, float full_ip) {
//     if (weight < 1e-6) return;

// 	float lam_dis = 0, dis2 = 0;

//     dis2 = 2 - 2 * full_ip;
//     if (dis2 < 1e-8) return;
//     lam_dis = 3 * sqrt(dis2) * weight;

//     for (int c = 0; c < dim; c++) {
// 	    vec_error_u[c] -= lam_dis * (vec_u[c] - tmp_chipped_layer_ip * vec_v[c]);
// 	    vec_error_v[c] -= lam_dis * (vec_v[c] - tmp_chipped_layer_ip * vec_u[c]);
// 	}
// }

inline float Backward_xx(long long u, long long v, 
    float *vec_error_u, float *vec_error_v, float *layer_weight_error_u, float *layer_weight_error_v, 
    float *tmp_column_inner_prod, float *tmp_layer_inner_prod, 
    int label, bool logsigmoid) {
    float tmp_unit_ip = 0;
    float tmp_layer_ip = 1;
    float tmp_chipped_layer_ip = 1;
    float *vec_u, *vec_v;
    float score = 0.;
    float ip = 0.;
    float lw1 = 0., lw2 = 0.;

    for (int l1 = 0; l1 < L; l1++) 
    for (int l2 = 0; l2 < L; l2++) {
        tmp_layer_ip = 1.;
        for (int c = 0; c < C; c++) {
            // cout << "column " << c << endl;
            vec_u = GetEmbVertex(u, l1, c, 0);
            // PrintEmb(vec_u, dim);
            vec_v = GetEmbVertex(v, l2, c, arch);
            tmp_unit_ip = DotProd(vec_u, vec_v, dim);
            tmp_column_inner_prod[l1 * L * C + l2 * C + c] = tmp_unit_ip;
            tmp_layer_ip *= tmp_unit_ip;

            if (constrain_zero && tmp_unit_ip < 0) for (int d = 0; d < dim; d++) {
                vec_error_u[l1 * C * dim + c * dim + d] += vec_v[d];
                vec_error_v[l2 * C * dim + c * dim + d] += vec_u[d];
            }
        }
        tmp_layer_inner_prod[l1 * L + l2] = tmp_layer_ip;
    }

    ip = 0;
    if (use_lw) for (int l1 = 0; l1 < L; l1++) {
        lw1 = layer_weight.GetWeight(u, l1);
        for (int l2 = 0; l2 < L; l2++) {
            lw2 = layer_weight.GetWeight(v, l2);
            ip += tmp_layer_inner_prod[l1 * L + l2] * (lw1 * lw1 * lw2 * lw2);
        }
    }
    else for (int l1 = 0; l1 < L; l1++) for (int l2 = 0; l2 < L; l2++) 
        ip += tmp_layer_inner_prod[l1 * L + l2] / (L * L);

    if (constrain_zero && logsigmoid) score = label - FastSigmoid((ip - 0.5) / temp);
    else 
    if (logsigmoid) score = label - FastSigmoid(ip  / temp);
    else score = 2 * (label - 0.5);

    float w;
    for (int l1 = 0; l1 < L; l1 ++)
    for (int l2 = 0; l2 < L; l2++) 
    for (int c = 0; c < C; c++) {
        vec_u = GetEmbVertex(u, l1, c, 0);
        vec_v = GetEmbVertex(v, l2, c, arch);
        if (vec_u == vec_v) continue;
        // get chipped ip
        tmp_chipped_layer_ip = 1.;
        for (int c2 = 0; c2 < C; c2++) 
        if (c2 != c)
            tmp_chipped_layer_ip *= tmp_column_inner_prod[l1 * L * C + l2 * C + c2];
        if (use_lw) {
            lw1 = layer_weight.GetWeight(u, l1);
            lw2 = layer_weight.GetWeight(v, l2);
            w = tmp_chipped_layer_ip * (lw1 * lw1 * lw2 * lw2);
        }
        else w = tmp_chipped_layer_ip / (L * L);
        
        for (int d = 0; d < dim; d++) {
            vec_error_u[l1 * C * dim + c * dim + d] += score * w * vec_v[d] / temp;
            vec_error_v[l2 * C * dim + c * dim + d] += score * w * vec_u[d] / temp;
        }
        if (label) 
            // BackwardReg(vec_u, vec_v, &vec_error_u[l1 * C * dim + c * dim], &vec_error_v[l2 * C * dim + c * dim], tmp_chipped_layer_ip, regw, ip);
            BackwardReg(vec_u, vec_v, &vec_error_u[l1 * C * dim + c * dim], &vec_error_v[l2 * C * dim + c * dim], regw);
    }
    if (use_lw) for (int l1 = 0; l1 < L; l1 ++) {
        lw1 = layer_weight.GetWeight(u, l1);
        for (int l2 = 0; l2 < L; l2++) {
            lw2 = layer_weight.GetWeight(v, l2);
            layer_weight_error_u[l1] += score * lw2 * lw2 * tmp_layer_inner_prod[l1 * L + l2] * 2 * lw1;
            layer_weight_error_v[l2] += score * lw1 * lw1 * tmp_layer_inner_prod[l1 * L + l2] * 2 * lw2;
        }
    }
    
    if (logsigmoid) return score;
    else return score * ip;
}

inline float UpdateMT(int u, int v, int negv,
    float *vec_error_u, float *vec_error_v, float *vec_error_negv, 
    float *layer_weight_error_u, float *layer_weight_error_v, float *layer_weight_error_negv, 
    float *tmp_column_inner_prod, float *tmp_layer_inner_prod) {
    
    if (u == negv || u == v) return 0;
    if (net_flag && net.HasEdge(u, negv)) return 0;

    float posscore, negscore, loss;

    // update vec_u, vec_v
    posscore = Backward_xx(u, v, vec_error_u, vec_error_v, 
        layer_weight_error_u, layer_weight_error_v, 
        tmp_column_inner_prod, tmp_layer_inner_prod, 1, false);
    negscore = Backward_xx(u, negv, vec_error_u, vec_error_negv, 
        layer_weight_error_u, layer_weight_error_negv, 
        tmp_column_inner_prod, tmp_layer_inner_prod, 0, false);
    
    loss = posscore + negscore - temp * L * margin;
    if (loss < 0) {
        // update embedding
        for (int l = 0; l < L; l++) for (int c = 0; c < C; c++) {
            switch (opt) {
            case 1:
                embedding.UpdateEmbSGD(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho);
                embedding.UpdateEmbSGD(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho);
                embedding.UpdateEmbSGD(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho);
                break;
            case 2:
                embedding.UpdateEmbBSGD(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho, batch_size, riemann_order);
                embedding.UpdateEmbBSGD(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho, batch_size, riemann_order);
                embedding.UpdateEmbBSGD(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho, batch_size, riemann_order);
                break;
            case 3:
                embedding.UpdateEmbAdagrad(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho, batch_size, false, riemann_order);
                embedding.UpdateEmbAdagrad(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho, batch_size, false, riemann_order);
                embedding.UpdateEmbAdagrad(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho, batch_size, false, riemann_order);
                break;
            case 4:
                embedding.UpdateEmbAdagrad(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho, batch_size, true, riemann_order);
                embedding.UpdateEmbAdagrad(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho, batch_size, true, riemann_order);
                embedding.UpdateEmbAdagrad(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho, batch_size, true, riemann_order);
                break;
            default:
                cout << "ERROR: opt=" << opt << " is invalid!\n";
                exit(1);
            }
            if (norm_sphere) {
                Normalize(GetEmbVertex(u, l, c, 0), dim, 1.);
                Normalize(GetEmbVertex(v, l, c, arch), dim, 1.);
                Normalize(GetEmbVertex(negv, l, c, arch), dim, 1.);
            }
        }
        // update layer weights
        if (use_lw) {
            layer_weight.UpdateWeightSGD(u, layer_weight_error_u, rho / batch_size);
            Normalize(layer_weight.GetWeightPtr(u), L);
            layer_weight.UpdateWeightSGD(v, layer_weight_error_v, rho / batch_size);
            Normalize(layer_weight.GetWeightPtr(v), L);
            layer_weight.UpdateWeightSGD(v, layer_weight_error_negv, rho / batch_size);
            Normalize(layer_weight.GetWeightPtr(negv), L);
        }
    }
    Fill(vec_error_u, L * C * dim, 0);
    Fill(vec_error_v, L * C * dim, 0);
    Fill(vec_error_negv, L * C * dim, 0);
    if (use_lw) {
        Fill(layer_weight_error_u, L, 0);
        Fill(layer_weight_error_v, L, 0);
        Fill(layer_weight_error_negv, L, 0);
    }

    return loss > 0 ? 0 : loss;
}

inline float UpdateSigmoid(int u, int v,
    float *vec_error_u, float *vec_error_v, float *vec_error_negv, 
    float *layer_weight_error_u, float *layer_weight_error_v, float *layer_weight_error_negv,
    float *tmp_column_inner_prod, float *tmp_layer_inner_prod) {

    // int negv;
    float posscore, negscore, loss = 0;

    // bp for u, v
    posscore = Backward_xx(u, v, vec_error_u, vec_error_v, 
        layer_weight_error_u, layer_weight_error_v, 
        tmp_column_inner_prod, tmp_layer_inner_prod, 1, true);
    loss += posscore;
    
    for (int n = 0, negv = net_flag ? net.SampleNeg() : seq.SampleNeg(); 
        n < num_neg; 
        n++, negv = net_flag ? net.SampleNeg() : seq.SampleNeg()) {
        // bp for u, negv
        negscore = Backward_xx(u, negv, vec_error_u, vec_error_negv, 
            layer_weight_error_u, layer_weight_error_negv, 
            tmp_column_inner_prod, tmp_layer_inner_prod, 0, true);
        loss += negscore;
        // update embedding of negv
        for (int l = 0; l < L; l++) for (int c = 0; c < C; c++) {
            switch (opt) {
            case 1:
                embedding.UpdateEmbSGD(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho);
                break;
            case 2:
                embedding.UpdateEmbBSGD(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho, batch_size, riemann_order);
                break;
            case 3:
                embedding.UpdateEmbAdagrad(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho, batch_size, false, riemann_order);
                break;
            case 4:
                embedding.UpdateEmbAdagrad(l, GetRowIdx(negv, l, c), c, &vec_error_negv[l * C * dim + c * dim], arch, dim, rho, batch_size, true, riemann_order);
                break;
            default:
                cout << "ERROR: opt=" << opt << " is invalid!\n";
                exit(1);
            }
            if (norm_sphere) Normalize(GetEmbVertex(negv, l, c, arch), dim, 1.);
        }
        Fill(vec_error_negv, L * C * dim, 0);
        // update layer weights
        if (use_lw) {
            layer_weight.UpdateWeightSGD(negv, layer_weight_error_negv, rho / batch_size);
            Normalize(layer_weight.GetWeightPtr(negv), L);
            Fill(layer_weight_error_negv, L, 0);
        }
    }

    // update embedding of  u, v
    for (int l = 0; l < L; l++) for (int c = 0; c < C; c++) {
        switch (opt) {
        case 1:
            embedding.UpdateEmbSGD(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho);
            embedding.UpdateEmbSGD(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho);
            break;
        case 2:
            embedding.UpdateEmbBSGD(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho, batch_size, riemann_order);
            embedding.UpdateEmbBSGD(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho, batch_size, riemann_order);
            break;
        case 3:
            embedding.UpdateEmbAdagrad(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho, batch_size, false, riemann_order);
            embedding.UpdateEmbAdagrad(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho, batch_size, false, riemann_order);
            break;
        case 4:
            embedding.UpdateEmbAdagrad(l, GetRowIdx(u, l, c),    c, &vec_error_u[l * C * dim + c * dim], 0, dim, rho, batch_size, true, riemann_order);
            embedding.UpdateEmbAdagrad(l, GetRowIdx(v, l, c),    c, &vec_error_v[l * C * dim + c * dim], arch, dim, rho, batch_size, true, riemann_order);
            break;
        default:
            cout << "ERROR: opt=" << opt << " is invalid!\n";
            exit(1);
        }
        if (norm_sphere) {
            Normalize(GetEmbVertex(u, l, c, 0), dim, 1.);
            Normalize(GetEmbVertex(v, l, c, arch), dim, 1.);
        }
    }
    Fill(vec_error_u, L * C * dim, 0);
    Fill(vec_error_v, L * C * dim, 0);
    if (use_lw) {
        layer_weight.UpdateWeightSGD(u, layer_weight_error_u, rho / batch_size);
        Normalize(layer_weight.GetWeightPtr(u), L);
        Fill(layer_weight_error_u, L, 0);
        layer_weight.UpdateWeightSGD(v, layer_weight_error_v, rho / batch_size);
        Normalize(layer_weight.GetWeightPtr(v), L);
        Fill(layer_weight_error_v, L, 0);
    }

    return loss;
}

void *TrainThread(void *id)
{
    int tid = *(int *)id;
    // cout << "Create thread " << tid << endl;
    NodeSequences seq_thread;
    if (seq_flag) seq_thread.InitFile(sequence_file, window_size, tid, num_threads);
    

    int u, v, negv;
    long long count = 0, last_count = 0, curedge;
    // unsigned long long seed = (long long)id;
    FILE *logf;
    if (log_loss) logf = fopen(log_file, "wb");
    float loss, tmp_loss;

    float *vec_error_u = (float *)calloc(L * C * dim, sizeof(float));
    Fill(vec_error_u, L * C * dim, 0);
    float *vec_error_v = (float *)calloc(L * C * dim, sizeof(float));
    Fill(vec_error_v, L * C * dim, 0);
    float *vec_error_negv = (float *)calloc(L * C * dim, sizeof(float));
    Fill(vec_error_negv, L * C * dim, 0);

    float *layer_weight_error_u = (float *)calloc(L, sizeof(float));
    Fill(layer_weight_error_u, L, 0);
    float *layer_weight_error_v = (float *)calloc(L, sizeof(float));
    Fill(layer_weight_error_v, L, 0);
    float *layer_weight_error_negv = (float *)calloc(L, sizeof(float));
    Fill(layer_weight_error_negv, L, 0);

    float *tmp_column_inner_prod = (float *)calloc(L * L * C, sizeof(float));
    Fill(tmp_column_inner_prod, L * L * C, 0);
    float *tmp_layer_inner_prod = (float *)calloc(L * L, sizeof(float));
    Fill(tmp_layer_inner_prod, L * L, 0);

    while (1)
    {
        if (count > total_samples / num_threads + 2) break;

        if (count - last_count > 10000)
        {
            current_sample_count += count - last_count;
            last_count = count;
            
            if (print_progress) {
                printf("%cRho: %f  Progress: %.3lf%%    ", 13, rho, (float)current_sample_count / (float)(total_samples + 1) * 100);
                fflush(stdout);
            }

            if (opt < 3) {
                rho = init_rho * (1 - current_sample_count / (float)(total_samples + 1));
                if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
            }
        }


        if (net_flag) {
            curedge = net.SampleEdge();
            u = net.edge_source_id[curedge];
            v = net.edge_target_id[curedge];
            // u = net.SampleNode();
            // v = net.NextNode(u);
        }
        if (seq_flag) {
            seq_thread.NextPair(num_threads);
            u = seq_thread.pos_node_pair[0];
            v = seq_thread.pos_node_pair[1];
            // cout << u << " " << v << endl;
        }
        
        // loss = 0.;
        tmp_loss = 0.;


        if (net_flag) {
        if (rw) for (int w = 0; w < window_size; w++) {
            if (rand() / (float)RAND_MAX > 1. / (w + 1)) continue;
            if (w > 0) v = net.NextNode(v);

            if (obj == 0) {
                tmp_loss = UpdateSigmoid(u, v,
                    vec_error_u, vec_error_v, vec_error_negv,
                    layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                    tmp_column_inner_prod, tmp_layer_inner_prod);
                count++;
                loss += tmp_loss;
                if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
            }
            if (obj == 1) {
                negv = net.SampleNeg();
                tmp_loss = UpdateMT(u, v, negv, 
                    vec_error_u, vec_error_v, vec_error_negv,
                    layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                    tmp_column_inner_prod, tmp_layer_inner_prod);
                count++;
                loss += tmp_loss;
                if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
            }
            if (obj == 2) {
                if (rand() / (float)RAND_MAX < obj_mix_thresh) {
                    tmp_loss = UpdateSigmoid(u, v,
                        vec_error_u, vec_error_v, vec_error_negv,
                        layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                        tmp_column_inner_prod, tmp_layer_inner_prod);
                    count++;
                    if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
                } else {
                    negv = net.SampleNeg();
                    tmp_loss = UpdateMT(u, v, negv, 
                        vec_error_u, vec_error_v, vec_error_negv,
                        layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                        tmp_column_inner_prod, tmp_layer_inner_prod);
                    count++;
                    loss += tmp_loss;
                    if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
                }
            }
        } else if (rwr) {
            v = net.NextNodeR(u, ppralpha);
            if (obj == 0) {
                tmp_loss = UpdateSigmoid(u, v,
                    vec_error_u, vec_error_v, vec_error_negv,
                    layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                    tmp_column_inner_prod, tmp_layer_inner_prod);
                count++;
                loss += tmp_loss;
                if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
            }
            if (obj == 1) {
                negv = net.SampleNeg();
                tmp_loss = UpdateMT(u, v, negv, 
                    vec_error_u, vec_error_v, vec_error_negv,
                    layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                    tmp_column_inner_prod, tmp_layer_inner_prod);
                count++;
                loss += tmp_loss;
                if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
            }
            if (obj == 2) {
                if (rand() / (float)RAND_MAX < obj_mix_thresh) {
                    tmp_loss = UpdateSigmoid(u, v,
                        vec_error_u, vec_error_v, vec_error_negv,
                        layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                        tmp_column_inner_prod, tmp_layer_inner_prod);
                    count++;
                    loss += tmp_loss;
                    if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
                } else {
                    negv = net.SampleNeg();
                    tmp_loss = UpdateMT(u, v, negv, 
                        vec_error_u, vec_error_v, vec_error_negv,
                        layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                        tmp_column_inner_prod, tmp_layer_inner_prod);
                    count++;
                    loss += tmp_loss;
                    if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
                }
            }
        }
        }

        if (seq_flag) {
        if (obj == 0) {
            tmp_loss = UpdateSigmoid(u, v,
                vec_error_u, vec_error_v, vec_error_negv,
                layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                tmp_column_inner_prod, tmp_layer_inner_prod);
            count++;
            loss += tmp_loss;
            if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
        }
        if (obj == 1) {
            negv = seq.SampleNeg();
            tmp_loss = UpdateMT(u, v, negv, 
                vec_error_u, vec_error_v, vec_error_negv,
                layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                tmp_column_inner_prod, tmp_layer_inner_prod);
            count++;
            loss += tmp_loss;
            if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
        }
        if (obj == 2) {
            if (rand() / (float)RAND_MAX < obj_mix_thresh) {
                tmp_loss = UpdateSigmoid(u, v,
                    vec_error_u, vec_error_v, vec_error_negv,
                    layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                    tmp_column_inner_prod, tmp_layer_inner_prod);
                count++;
                loss += tmp_loss;
                if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
            } else {
                negv = seq.SampleNeg();
                tmp_loss = UpdateMT(u, v, negv, 
                    vec_error_u, vec_error_v, vec_error_negv,
                    layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
                    tmp_column_inner_prod, tmp_layer_inner_prod);
                count++;
                loss += tmp_loss;
                if (log_loss && count % 10 == 0) {fprintf(logf, "%f\n", loss / 10); loss = 0;}
            }
        }
        }
        // else for (int w = 0; w < window_size; w++) {
        //     // the first step RW(u), the second step RWR(v)
        //     if (w > 0) v = net.NextNodeR(v, ppralpha);
        //     negv = net.SampleNeg();
        //     UpdateMT(u, v, negv, 
        //         vec_error_u, vec_error_v, vec_error_negv,
        //         layer_weight_error_u, layer_weight_error_v, layer_weight_error_negv,
        //         tmp_column_inner_prod, tmp_layer_inner_prod);
        //     count++;
        //     if (log_loss && count % 10 == 0) fprintf(logf, "%f\n", loss);
        // }
    }

    if (log_loss) fclose(logf);

// free all
    free(vec_error_u);
    free(vec_error_v);
    free(tmp_layer_inner_prod);
    free(tmp_column_inner_prod);
    if (use_lw) {
        free(layer_weight_error_u);
        free(layer_weight_error_v);
        free(layer_weight_error_negv);
    }
    pthread_exit(NULL);
    // return NULL;
}

bool Bigger(float num1, float num2) {
    return num1 > num2;
}

float GetInnerProduct(int u, int v) {
    float score = 0.;
    float tmp_layer_ip = 1;

    for (int l1 = 0; l1 < L; l1++) for (int l2 = 0; l2 < L; l2++) {
        tmp_layer_ip = 1.;
        for (int c = 0; c < C; c++) tmp_layer_ip *= DotProd(
            GetEmbVertex(u, l1, c, 0), GetEmbVertex(v, l2, c, 0), dim);
        if (use_lw) score += layer_weight.GetWeight(u, l1) * layer_weight.GetWeight(u, l1)
            * layer_weight.GetWeight(v, l2) * layer_weight.GetWeight(v, l2) * tmp_layer_ip;
        else score += tmp_layer_ip / (L * L);
    }
    return score;
}

void EvalNetworkReconstruction() {
    if (!net_flag) {printf("No networks, skip network reconstruction \n"); return;}

    // evaluate network reconstruction precision by scoring over all the pairs
    long long u, v;
    long long hits = 0;

    vector<float> scores((long long)net.num_memblock * net.num_memblock, -99999.);
    for (long long u = 0; u < net.num_memblock; u++) for (long long v = 0; v < net.num_memblock; v++) {
        if (u == v) continue;
        scores[u * net.num_memblock + v] = GetInnerProduct(u, v);
        // cout << u << " " << v << " " << scores[u * net.num_memblock + v] <<endl;
    }
    vector<float> sorted_scores(scores);
    sort(sorted_scores.begin(), sorted_scores.end(), Bigger);
    float thresh = sorted_scores[net.num_edges];
    printf(
        "Min score: %f, threshold score: %f, max score: %f\n",
        sorted_scores[net.num_memblock * (net.num_memblock - 1) - 1],
        thresh,
        sorted_scores[0]);
    for (long long i = 0; i < net.num_edges; i++) {
        u = net.edge_source_id[i];
        v = net.edge_target_id[i];
        if (u == v) continue;
        if (scores[u * net.num_memblock + v] > thresh) hits++;
    }
    printf("Network reconstruction precision: %lf\n", (double)hits / net.num_edges);
}

void EstimatedEvalNetworkReconstruction(int nsr) {
    if (!net_flag) {printf("No networks, skip network reconstruction \n"); return;}

    srand(20000112);

    // evaluate network reconstruction precision by sampling
    long long u, v;
    float score;
    long long hits = 0;
    long long valid_counts = 0;

    vector<float> scores;
    for (long long i = 0; i < net.num_edges; i++) {
        u = net.edge_source_id[i];
        v = net.edge_target_id[i];
        if (u == v) continue;
        score = GetInnerProduct(u, v);
        scores.push_back(score);
        valid_counts++;
    }
    // sampling
    for (long long i = 0; i < num_edges * (nsr - 1); i++) {
        u = net.num_memblock * rand() / RAND_MAX;
        v = net.num_memblock * rand() / RAND_MAX;
        score = GetInnerProduct(u, v);
        scores.push_back(score);
    }

    vector<float> sorted_scores(scores);
    sort(sorted_scores.begin(), sorted_scores.end(), Bigger);
    float thresh = sorted_scores[valid_counts - 1];
    printf(
        "Min score: %f, threshold score: %f, max score: %f\n",
        sorted_scores[sorted_scores.size() - 1],
        thresh,
        sorted_scores[0]);
    for (long long i = 0; i < net.num_edges; i++) {
        u = net.edge_source_id[i];
        v = net.edge_target_id[i];
        if (u == v) continue;
        if (scores[i] > thresh) hits++;
    }
    printf("(Estimated) network reconstruction precision: %lf\n", (double)hits / valid_counts);
}

void OutputEmbedding(char *file)
{
    printf("Writiing Node Embedding to file %s\n", file);
    
    FILE *fo;
    double fd = 1.;
    for (int c = 0; c < C; c++) fd *= dim;
    if (fd > 4097) {
        printf("Full dim (%.0lf) is too large, stop writing node embedding.\n", fd);
        return;
    }

    int full_dim = (int)fast_pow(dim, C);

    float *emb = new float [full_dim];
    // writing embedding file
    fo = fopen(file, "wb");
    fprintf(fo, "%lld %d\n", net_flag ? net.num_memblock : seq.num_memblock, full_dim);
    for (int a = 0; a < (net_flag ? net.num_memblock : seq.num_memblock); a++) {
        fprintf(fo, "%d ", a);
        GetFullEmb(a, emb);
        for (int b = 0; b < full_dim; b++) fprintf(fo, "%.12f ", emb[b]);                
        fprintf(fo, "\n");
    }
    delete [] emb;
    fclose(fo);
}

void PrintParameters()
{
    printf("Debug mode: %d\n", debug);
    printf("-----File Settings---------------------------------------\n");
    if (net_flag = strcmp(network_file, "None")) printf("Network file: %s \n", network_file);
    if (seq_flag = strcmp(sequence_file, "None")) printf("Sequence file: %s \n", sequence_file);
    if (strcmp(sub_emb_config_file, "None")) {printf("sub_emb config file: %s \n", sub_emb_config_file); readtu_flag=true;}
    printf("Node embedding: %s \n", node_embedding_file);
    printf("TU embedding src: %s \n", sub_embedding_file_src);
    //if (arch == 1) printf("Emb-v:\t%s \n", v_embedding_file);
    printf("-----Training Settings-----------------------------------\n");
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Regw: %.3f\n", regw);
    printf("Temperature: %.3f\n", temp);

    printf("Objective: %s\t", obj_str);
    if (!strcmp(obj_str, "mt")) obj = 1;
    else if (!strcmp(obj_str, "sgns")) obj = 0;
    else if (!strcmp(obj_str, "mix")) obj = 2;
    else {printf("ERROR: obj is invalid.\n"); exit(1);}

    if (obj == 2) printf("%.2f sgns + %.2f mt\n", obj_mix_thresh, 1 - obj_mix_thresh);
    if (obj == 0 || obj == 2) printf("Num negative: %d\n", num_neg);
    if (obj == 1 || obj == 2) printf("Margin: %.3f\n", margin);

    if (L == 1 && use_lw == 1) {printf("WARNING: L = 1, set use_lw = 0\n"); use_lw = 0;}
    printf("Use layer weight: %d\n", use_lw);

    printf("Rw/Rwr: %d/%d\t", rw, rwr);
    if (rw == 0 && rwr == 0 && net_flag) {cout << "Sampling strategy should be initialized explicitly!" << endl; exit(1);}
    if (rw) printf("Window size: %d\n", window_size);
    if (rwr) printf("PPRalpha: %.3f\n", ppralpha);
    printf("Initial rho: %.3lf\n", init_rho);

    printf("Optimizer: %s\n", opt_str);
    if (!strcmp(opt_str, "sgd")) opt = 1;
    else if (!strcmp(opt_str, "bsgd")) opt = 2;
    else if (!strcmp(opt_str, "adagrad")) opt = 3;
    else if (!strcmp(opt_str, "rmsprop")) opt = 4;
    else {printf("ERROR: opt is invalid.\n"); exit(1);}
    if (opt > 1) printf("Batch size: %d\n", batch_size);

    printf("Riemann order: %d\n", riemann_order);
    printf("Sphere norm: %d\n", norm_sphere);
    printf("Zero constrain: %d\n", constrain_zero);

    printf("Threads: %d\t", num_threads);
    printf("\n-----Tensor Embedding Network Settings-------------------\n");
    printf("L: %d\tC: %d\tdim: %d\t", L, C, dim);
    printf("NCE type: %d\n", arch);
    printf("---------------------------------------------------------\n");

    // if (rw + rwr == 0) {cout << "Random walk should be initialized!\n"; exit(1);}
}

void Train() 
{
    long a;
    pthread_t *pt; // = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    if (net_flag) net.InitNetwork(network_file);
    else if (seq_flag) seq.InitFile(sequence_file, window_size, 0, 1);
    else {printf("ERROR: network_file or sequence_file should be initialized\n"); exit(1);}
    if (readtu_flag) tutab.ReadTUConfig(sub_emb_config_file, net_flag ? (int)net.num_memblock : (int)seq.num_memblock);
    if (readtu_flag) embedding.InitTensorizedEmbedding(L, C, tutab.R, dim, 0., arch, opt_str);
    else embedding.InitTensorizedEmbedding(L, C, (net_flag ? net.num_memblock : seq.num_memblock), dim, 0., arch, opt_str);
    if (obj == 0 || obj == 2) InitSigmoidTable();
    if (use_lw) layer_weight.InitLayerWeight((net_flag ? net.num_memblock : seq.num_memblock), L, false);

    clock_t start = clock();
    printf("--------------------------------\n");
    
    pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    int *pids = new int [num_threads]; 
    for (a = 0; a < num_threads; a++) {pids[a] = a; pthread_create(&pt[a], NULL, TrainThread, &pids[a]);}
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf s\n", (double)(finish - start) / CLOCKS_PER_SEC / num_threads);

    // evaluation
    if (eval_nr == 1) EvalNetworkReconstruction();
    // if (eval_nr == 2) {
    //     EstimatedEvalNetworkReconstruction(2);
    //     EstimatedEvalNetworkReconstruction(10);
    //     EstimatedEvalNetworkReconstruction(50);
    // }

    if (use_lw) layer_weight.OutputWeight(layer_weight_file);
    embedding.OutputTensorEmbedding(sub_embedding_file_src, 0., tutab);
    OutputEmbedding(node_embedding_file);
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) 
{
    int i;
    printf("==============node2ket==============\n");
    if ((i = ArgPos((char *)"-net", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-seq", argc, argv)) > 0) strcpy(sequence_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-config", argc, argv)) > 0) strcpy(TU_config_file, argv[i + 1]);
    
    if ((i = ArgPos((char *)"-mt-mar", argc, argv)) > 0) margin = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-temp", argc, argv)) > 0) temp = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-num-neg", argc, argv)) > 0) num_neg = atoi(argv[i + 1]);

    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = (long long)(1000000 * atof(argv[i + 1]));
    if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-regw", argc, argv)) > 0) regw = atof(argv[i + 1]);

    if ((i = ArgPos((char *)"-type", argc, argv)) > 0) arch = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-C", argc, argv)) > 0) C = atoi(argv[i + 1]); // order
    if ((i = ArgPos((char *)"-L", argc, argv)) > 0) L = atoi(argv[i + 1]); // rank
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-seed", argc, argv)) > 0) seed = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-use-lw", argc, argv)) > 0) use_lw = atoi(argv[i + 1]);
    
    if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-log", argc, argv)) > 0) log_loss = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-print", argc, argv)) > 0) print_progress = atoi(argv[i + 1]);

    if ((i = ArgPos((char *)"-rw", argc, argv)) > 0) rw = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rwr", argc, argv)) > 0) rwr = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-window-size", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-ppralpha", argc, argv)) > 0) ppralpha = atof(argv[i + 1]);

    if ((i = ArgPos((char *)"-eval-nr", argc, argv)) > 0) eval_nr = atoi(argv[i + 1]);

    if ((i = ArgPos((char *)"-outputemb", argc, argv)) > 0) outputemb = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-node-emb", argc, argv)) > 0) strcpy(node_embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-TU-emb-src", argc, argv)) > 0) strcpy(sub_embedding_file_src, argv[i + 1]);

    if ((i = ArgPos((char *)"-opt", argc, argv)) > 0) strcpy(opt_str, argv[i + 1]);
    if ((i = ArgPos((char *)"-obj", argc, argv)) > 0) strcpy(obj_str, argv[i + 1]);
    if ((i = ArgPos((char *)"-obj-thresh", argc, argv)) > 0) obj_mix_thresh = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-norm", argc, argv)) > 0) norm_sphere = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-zero", argc, argv)) > 0) constrain_zero = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-riemann", argc, argv)) > 0) riemann_order = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-batch-size", argc, argv)) > 0) batch_size = atoi(argv[i + 1]);
    
    rho = init_rho;

    PrintParameters();
    cout << "Random seed: " << seed << endl;
    rng.seed(seed);
    Train();

    // int pid = getpid();
    // system

    return 0;
}
