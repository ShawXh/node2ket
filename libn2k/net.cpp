#include "net.h"

const int NEG_TABLE_SIZE = 1e8;
const double NEG_SAMPLING_POWER = 0.75;
const int MAX_NODE_NUMBER_ADJ = 10000;
const int MAX_STRING = 1024;

std::mt19937 rng(time(0));
std::uniform_real_distribution<double> randf(0., 1.);

void AliasTable::InitAliasTable(long long num_edges, float *edge_weight) {
    printf("Initializing Alias Table...\n");
    this->num_items = num_edges;
    this->alias = (long long *)malloc(num_edges*sizeof(long long));
    this->prob = (double *)malloc(num_edges*sizeof(double));
    if (this->alias == NULL || this->prob == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double *norm_prob = (double*)malloc(num_edges*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
    for (long long k = 0; k != num_edges; k++) 
        norm_prob[k] = edge_weight[k] * num_edges / sum;
    
    // init alias map
    for (long long k = num_edges - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        this->prob[cur_small_block] = norm_prob[cur_small_block];
        this->alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) this->prob[large_block[--num_large_block]] = 1;
    while (num_small_block) this->prob[small_block[--num_small_block]] = 1;

    free(norm_prob);
    free(small_block);
    free(large_block);
}

long long AliasTable::Sample() {
    float rand_value1 = randf(rng);
    float rand_value2 = randf(rng);
    long long k = (long long)this->num_items * rand_value1;
    return rand_value2 < this->prob[k] ? k : this->alias[k];
}

void Network::InitNetwork(const char *path){
    strcpy(this->path, path);
    printf("Read network from %s\n", this->path);
    ReadData();
    InitNegTable();
    this->alt.InitAliasTable(this->num_edges, this->edge_weight);
};

void Network::ReadData() {
    FILE *fin;
    char str[300];
    float weight;
    int vid1, vid2;
	int max_vid = 0;

    char *network_file = this->path;
    fin = fopen(network_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file %s not found!\n", network_file);
        exit(1);
    }
    this->num_edges = 0;
    while (fgets(str, sizeof(str), fin)) this->num_edges++;
    fclose(fin);
    printf("Number of edges: %lld\n", this->num_edges);

    this->edge_source_id = (int *)malloc(this->num_edges * sizeof(int));
    this->edge_target_id = (int *)malloc(this->num_edges * sizeof(int));
    this->edge_weight = (float *)malloc(this->num_edges * sizeof(float));
    if (this->edge_source_id == NULL || this->edge_target_id == NULL || this->edge_weight == NULL) {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    // read num vertices and edges
    fin = fopen(network_file, "rb");
    for (int k = 0; k != this->num_edges; k++)
    {
        fscanf(fin, "%d %d %f\n", &vid1, &vid2, &weight);
        this->edge_source_id[k] = vid1;
        this->edge_target_id[k] = vid2;
        this->edge_weight[k] = weight;
		max_vid = max_vid > vid1 ? max_vid: vid1;
		max_vid = max_vid > vid2 ? max_vid: vid2;
    }
	this->num_memblock = max_vid + 1;
    printf("Numer of nodes (max_node_id + 1): %lld\n", this->num_memblock);
    fclose(fin);

    // initialize adjacency matrix
    if (this->num_memblock < MAX_NODE_NUMBER_ADJ) {
        fin = fopen(network_file, "rb");
        this->adj_indicator = new bool [(long long)this->num_memblock * this->num_memblock];
        for (long long k = 0; k != this->num_memblock * this->num_memblock; k++)
        this->adj_indicator[k] = false;

        for (int k = 0; k != this->num_edges; k++)
        {
            fscanf(fin, "%d %d %f\n", &vid1, &vid2, &weight);
            this->adj_indicator[(long long)this->num_memblock * vid1 + vid2] = true;
            this->adj_indicator[(long long)this->num_memblock * vid2 + vid1] = true;
        }
        fclose(fin);
        
        this->initialized_adj = true;
    } else this->initialized_adj = false;

    // read this->neighbor
    fin = fopen(network_file, "rb");
    this->neighbor = new std::vector<long long> [this->num_memblock];
    for (int k = 0; k != this->num_edges; k++)
    {
        fscanf(fin, "%d %d %f\n", &vid1, &vid2, &weight);
        this->neighbor[vid1].push_back(vid2);
    }
    fclose(fin);

    // read this->degree
    this->degree = (float *)malloc(this->num_memblock * sizeof(float));
    for (int i = 0; i < this->num_memblock; i++) this->degree[i] = 0;
    fin = fopen(network_file, "rb");
    for (int k = 0; k != this->num_edges; k++)
    {
        fscanf(fin, "%d %d %f\n", &vid1, &vid2, &weight);
		this->degree[vid1] += weight;
		this->degree[vid2] += weight;
    }
    fclose(fin);
}

bool Network::HasEdge(int i, int j) {
    if (this->initialized_adj) return this->adj_indicator[(long long) i * this->num_memblock + j];
    else for (int n = 0; n < (int)this->neighbor[i].size(); n++) {
        if (this->neighbor[i][n] == j) return true;
    }
    return false;
}

void Network::InitNegTable() {
    printf("Initializing Neg Table...\n");
    double sum=0, cur_sum = 0, por = 0;
    int vid = 0;
    this->neg_table = (int *)malloc(NEG_TABLE_SIZE * sizeof(int));
    if (this->neg_table == NULL) {printf("neg table malloc error.\n");exit(1);}

    for (int k = 0; k != this->num_memblock; k++)
        sum += pow(this->degree[k], NEG_SAMPLING_POWER);
    for (int k = 0; k != NEG_TABLE_SIZE; k++)
    {
        if ((double)(k + 1) / NEG_TABLE_SIZE > por)
        {
            cur_sum += pow(this->degree[vid], NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid ++;
        }
        if (vid >= this->num_memblock) this->neg_table[k] = this->num_memblock - 1;
        else this->neg_table[k] = vid - 1;
    }
}

void Network::SetRandomSeed(long long seed) {rng.seed(seed);}

long long Network::SampleEdge() {
    return this->alt.Sample();
}

int Network::NextNode(int u) {
    int idx = (int)(randf(rng) * this->neighbor[u].size());
    return neighbor[u][idx];
}

int Network::NextNodeR(int u, float ppralpha=0.85) {
    int n2 = u;
    while (randf(rng) < ppralpha) {
        int neighbor = this->NextNode(n2);
        n2 = neighbor;
    }
    return n2;
}

int Network::SampleNeg() {
    return this->neg_table[(long long)(randf(rng) * NEG_TABLE_SIZE)];
}

int Network::SampleNode() {
    return (int) (this->num_memblock * randf(rng));
}



void NodeSequences::InitFile(const char *path, int window_size, int id=0, int num_gap=1) {
    strcpy(this->path, path);
    this->cur_seq_idx = 0;
    this->id = id;
    this->cur_window_node_idx = 0;
    this->cur_center_node_idx = 0;
    this->pos_node_pair = new int [2];
    this->window_size = window_size;

    // cout << this->id << endl;

    char seq[MAX_STRING];

    int vid, max_vid = 0;

    this->file = fopen(this->path, "r");
    if (this->file == NULL) {
        printf("ERROR: open file %s\n", this->path);
        exit(1);
    }
    fgets(seq, MAX_STRING, this->file);
    for (int i = 0; i < MAX_STRING, seq[i+1] != 0; i++) {
        if (i == 0 && seq[i] >= '0' && seq[i] <= '9') this->seq_length++;
        else if (i > 0 && seq[i-1] == ' ' && seq[i] >= '0' && seq[i] <= '9') this->seq_length++;
    }
    this->num_sequences = 1;
    // cout << this->seq_length << endl;
    while (fgets(seq, MAX_STRING, this->file)) this->num_sequences++;
    
    fclose(this->file);
    this->cur_seq = (int *)malloc(this->seq_length * sizeof(int));

    //todo: might have problems
    this->file = fopen(this->path, "r");
    while(!feof(this->file)) {
        fscanf(this->file, "%d ", &vid);
        max_vid = vid > max_vid ? vid : max_vid;
    }
    this->num_memblock = max_vid + 1;
    
    printf("Loading sequences %lld sequences, seq_length %d, window size %d, number of nodes %lld\n", 
        this->num_sequences, this->seq_length, this->window_size, this->num_memblock);
    fclose(this->file);

    this->freq_count = (float *)malloc(this->num_memblock * sizeof(float));
    this->file = fopen(this->path, "r");
    this->cur_seq = (int *)malloc(this->seq_length * sizeof(int));
    this->freq_count = (float *)malloc(this->num_memblock * sizeof(float));
    for (long long i = 0; i < this->num_sequences; i++) {
        for (int w = 0; w < this->seq_length; w++) {
            fscanf(this->file, "%d", &vid);
            this->freq_count[vid] += 1;
        }
    }
    fclose(this->file);
    this->neg_node_alt.InitAliasTable(this->num_memblock, this->freq_count);

    this->file = fopen(this->path, "r");
    this->LoadSequence(num_gap);
}

void NodeSequences::NextPair(int num_gap = 0) {
    // cout << "current center node idx " << this->cur_center_node_idx << endl;
    // cout << "current window node idx " << this->cur_window_node_idx << endl;
    if (this->cur_window_node_idx == 0) {
        this->cur_window_node_idx++;
        return this->NextPair(num_gap);
    }
    if (this->cur_center_node_idx == this->seq_length) {
        this->LoadSequence(num_gap);
        this->cur_center_node_idx = 0;
        this->cur_window_node_idx = 1;
        return this->NextPair(num_gap);
    }
    if (this->cur_window_node_idx == this->window_size + 1){
        this->cur_center_node_idx++;
        this->cur_window_node_idx = -this->window_size;
        return this->NextPair(num_gap);
    }
    if (this->cur_center_node_idx + this->cur_window_node_idx < 0) {
        this->cur_window_node_idx = - this->cur_center_node_idx;
        return this->NextPair(num_gap);
    }
    if (this->cur_center_node_idx + this->cur_window_node_idx >= this->seq_length) {
        this->cur_center_node_idx++;
        this->cur_window_node_idx = -this->window_size;
        return this->NextPair(num_gap);
    }
    this->pos_node_pair[0] = this->cur_seq[cur_center_node_idx];
    this->pos_node_pair[1] = this->cur_seq[cur_center_node_idx + cur_window_node_idx];
    
    this->cur_window_node_idx++;
    return;
}

int NodeSequences::SampleNeg() {
    return (int)this->neg_node_alt.Sample();
    // return (int)(randf(rng) * this->num_memblock);
}

NodeSequences::~NodeSequences() {
    free(this->pos_node_pair);
    free(this->freq_count);
    // fclose(this->file);
}

void NodeSequences::LoadSequence(int num_gap) {
    while (this->cur_seq_idx % num_gap != this->id) {
        fgets(this->tmpseq, sizeof(this->tmpseq), this->file);
        this->cur_seq_idx++;
    }
    for (int i = 0; i < this->seq_length; i++) fscanf(this->file, "%d ", &this->cur_seq[i]);
}

void NodeSequences::Restart() {
    fclose(this->file);
    this->file = fopen(this->path, "r");
    fscanf(this->file, "%lld %lld %d", &this->num_memblock, &this->num_sequences, &this->seq_length);
}