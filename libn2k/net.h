#ifndef NET_H
#define NET_H

#include<cstring>
#include<iostream>
#include<vector>
#include<random>
#include<ctime>

using namespace std;

class TestExample {
public:
    TestExample() {};
    ~TestExample() {};
    void print1() {
        printf("1\n");
    }
};

class AliasTable {
public:
    long long *alias;
    double *prob;
    long long num_items;
    AliasTable() {};
    ~AliasTable() {};
    void InitAliasTable(long long num_items, float *item_weight);
    long long Sample();
};

class Network{
public: 
    char path[1024];
    int *neg_table;
    long long num_memblock;
    long long num_edges;
    int *edge_source_id, *edge_target_id;
    float *edge_weight; 
    vector<long long> *neighbor; 
    bool initialized_adj;
    bool *adj_indicator;
    float *degree;
    AliasTable alt;

    void InitNegTable();
    void ReadData();
    Network() {};   
    ~Network() {};
    
    void InitNetwork(const char *path);
    bool HasEdge(int i, int j);
    void SetRandomSeed(long long seed);
    long long SampleEdge();
    int SampleNeg();
    int SampleNode();
    int NextNode(int node);
    int NextNodeR(int node, float ppralpha);
    int * RandomWalk(int start_node);
};

class NodeSequences{
public:
    char path[1024];
    int *neg_table;
    long long num_memblock;
    FILE *file;

    int window_size;
    int seq_length;
    int *cur_seq;
    long long num_sequences;
    long long cur_seq_idx;
    int id;
    char tmpseq[1024];
    int cur_window_node_idx;
    int cur_center_node_idx;
    int *pos_node_pair;
    float *freq_count;
    AliasTable neg_node_alt;

    NodeSequences() {};
    ~NodeSequences();

    void InitFile(const char *path, int window_size, int id, int num_gap);
    void LoadSequence(int num_gap);
    void NextPair(int num_gap);
    int SampleNeg();

private:
    void Restart();
    
};

#endif