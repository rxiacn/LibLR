/********************************************************************
* Logit Model (Softmax Regression) V0.20
* Implemented by Rui Xia(rxiacn@gmail.com)
* Last updated on 2013-05-04. 
*********************************************************************/
#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <climits>
#include <math.h>
#include <time.h>
#include "lbfgs.h" 


# define VERSION       "V0.20"
# define VERSION_DATE  "2013-05-04"



using namespace std;

const long double LOG_LIM = 1e-300;


static int feat_set_size ;
static int class_set_size;
static vector< vector<double> > omega;



struct sparse_feat
{
	vector<int> id_vec;
	vector<float> value_vec;
};


class LR 
{
     
public:
    LR();
    ~LR();
	void save_model(string model_file);
    void load_model(string model_file);
	void load_training_file(string training_file);
	float classify_testing_file(string testing_file, string output_file, int output_format);
	void init_omega();
    int train_sgd(int max_loop, double loss_thrd, float learn_rate, float lambda);
	int train_gd(int max_loop, double loss_thrd, float learn_rate, float lambda);
	int train_lbfgs();
	void calc_loss(double *loss, float *acc);
	vector<sparse_feat> samp_feat_vec;
	vector<int> samp_class_vec;
	vector<double> calc_linear_sum(sparse_feat &samp_feat);
	vector<double> calc_softmax_prb(vector<double> &score_vec);
	static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    );
	static int LR::progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    );
	

	
private:
	void read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec);
    int score_to_class(vector<double> &score);
	float calc_acc(vector<int> &true_class_vec, vector<int> &pred_class_vec);
	vector<string> string_split(string terms_str, string spliting_tag);
	
	

};


