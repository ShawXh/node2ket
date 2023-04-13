#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <random>
#include <algorithm>

#include<emb.h>
#include<net.h>

#define MAX_STRING 1024

using namespace std;

typedef float real;                    // Precision of float numbers

char embedding_file[MAX_STRING], network_file[MAX_STRING];
char TU_config_file[MAX_STRING] = "None"; 
int vector_dim;
float *emb;
int nr_sample_rate = 10;
int tensorized_emb_flag = 0;

bool readtu_flag = false;
Network net;
TensorizedEmbedding tensor_emb;
TUTable tutab;

inline int GetRowIdx(int u, int l, int c) {
    if (readtu_flag) return tutab.row_table[(u * tensor_emb.L + l) * tensor_emb.C + c];
    else return u;
}

inline float * GetEmbVertex(int u, int l, int c, int fetch_h) {
    u = GetRowIdx(u, l, c);
    return tensor_emb.GetEmbx(l, u, c);
}

void ReadEmbedding()
{
	long long num_vertices, a, b;
    float *vec;
    long long vid;
	FILE *fi;

	fi = fopen(embedding_file, "rb");
    if (fi == NULL) {
        printf("Error: embedding file doesn't exit\n");
        exit(1);
    } 

	fscanf(fi, "%lld %d", &num_vertices, &vector_dim);
	printf("From embedding: num nodes: %lld, dim: %d\n", num_vertices, vector_dim);

	emb = (float *)malloc(net.num_memblock * vector_dim * sizeof(float));
	if (emb == NULL) {
		printf("ERROR: memory failed in emb\n");
		exit(1);
	}

    long long N = num_vertices;
    for (a = 0; a < N; a++)
	{
		fscanf(fi, "\n%lld", &vid);
		// cout << vid << endl;
		vec = &emb[vid * vector_dim];
		for (b = 0; b < vector_dim; b++) fscanf(fi, " %f", &vec[b]);
	}
	fclose(fi);
}

float GetInnerProduct(int u, int v) {
    float score = 0.;
    float tmp_layer_ip = 1;

    for (int l1 = 0; l1 < tensor_emb.L; l1++) for (int l2 = 0; l2 < tensor_emb.L; l2++) {
        tmp_layer_ip = 1.;
        for (int c = 0; c < tensor_emb.C; c++) tmp_layer_ip *= DotProd(
            GetEmbVertex(u, l1, c, 0), GetEmbVertex(v, l2, c, 0), tensor_emb.dim);
        // if (use_lw) score += layer_weight.GetWeight(u, l1) * layer_weight.GetWeight(u, l1)
        //     * layer_weight.GetWeight(v, l2) * layer_weight.GetWeight(v, l2) * tmp_layer_ip;
        // else 
        score += tmp_layer_ip / (tensor_emb.L * tensor_emb.L);
    }
    return score;
}

bool Bigger(float num1, float num2) {
    return num1 > num2;
}

float GetEuclideanDistance(long long u, long long v) {
	float score = 0.;
	float t = 0.;
    if (tensorized_emb_flag == 0) {
        for (int d = 0; d < vector_dim; d++) {
            t = emb[u * vector_dim + d] - emb[v * vector_dim + d];
            score += t * t;
        }
        return -sqrt(score);
    } else {
        return GetInnerProduct(u, v);
    }
}

void EstimatedEvalNetworkReconstruction(int nsr) {
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
        score = GetEuclideanDistance(u, v);
        scores.push_back(score);
        valid_counts++;
    }
    // sampling
    for (long long i = 0; i < net.num_edges * (nsr - 1); i++) {
        u = net.num_memblock * rand() / RAND_MAX;
        v = net.num_memblock * rand() / RAND_MAX;
        if (u == v || net.degree[u] < 0.5 || net.degree[v] < 0.5) {
            i--;
            continue;
        }
        score = GetEuclideanDistance(u, v);
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
    printf("(Estimated) network reconstruction precision: %lf, sample rate: %d\n", (double)hits / valid_counts, nsr);
}



int ArgPos(char *str, int argc, char **argv) {
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

int main(int argc, char **argv) {
	int i;
	if ((i = ArgPos((char *)"-emb", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-tensorized", argc, argv)) > 0) tensorized_emb_flag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-net", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-config", argc, argv)) > 0) strcpy(TU_config_file, argv[i + 1]);
    
    
    net.InitNetwork(network_file);
    if (strcmp(TU_config_file, "None")) {printf("TU config file: %s \n", TU_config_file); readtu_flag=true;}
    if (readtu_flag) tutab.ReadTUConfig(TU_config_file, (int)net.num_memblock);

    if (tensorized_emb_flag == 0) ReadEmbedding();
    else tensor_emb.LoadTensorEmbedding(embedding_file, tutab);

    // PrintEmb(tensor_emb.embx[0]+1000, 16);

	EstimatedEvalNetworkReconstruction(2);
    EstimatedEvalNetworkReconstruction(10);
    EstimatedEvalNetworkReconstruction(50);
	return 0;
}
