/********************************************************************
* Logit Model (Softmax Regression) V0.20
* Implemented by Rui Xia(rxiacn@gmail.com)
* Last updated on 2013-05-04.   
*********************************************************************/
#include "LR.h"
#include "lbfgs.h" 



static LR *p;
LR::LR()
{	
	p=this;
}

LR::~LR()
{
}




void LR::save_model(string model_file)
{ 
	cout << "Saving model..." << endl;
    ofstream fout(model_file.c_str());
	for (int j = 0; j < class_set_size - 1; j++) {
		for (int k = 0; k < feat_set_size; k++) {
			fout << omega[j][k] << " ";
		}
		fout << endl;
	}
    fout.close();
}


void LR::load_model(string model_file)
{
	cout << "Loading model..." << endl;
	omega.clear();
    ifstream fin(model_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << model_file << endl;
        exit(0);
    }    string line_str;
    while (getline(fin, line_str)) {
        vector<string> line_vec = string_split(line_str, " ");
        vector<double>  line_omega;
        for (vector<string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
			double weight = (double)atof(it->c_str());
			line_omega.push_back(weight);
		}
		omega.push_back(line_omega);
	}
	fin.close();
	class_set_size = (int)omega.size() + 1;
	feat_set_size = (int)omega[0].size();
}


void LR::read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec) {
    ifstream fin(samp_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << samp_file << endl;
        exit(0);
    }
    string line_str;
    while (getline(fin, line_str)) 
	{
        size_t class_pos = line_str.find_first_of("\t");
        int class_id = atoi(line_str.substr(0, class_pos).c_str());
        samp_class_vec.push_back(class_id);
        string terms_str = line_str.substr(class_pos+1);
        sparse_feat samp_feat;
        samp_feat.id_vec.push_back(0); // bias
        samp_feat.value_vec.push_back(1); // bias
        if (terms_str != "") 
		{
			vector<string> fv_vec = string_split(terms_str, " ");
			for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) 
			{
				size_t feat_pos = it->find_first_of(":");
				int feat_id = atoi(it->substr(0, feat_pos).c_str());
				float feat_value = (float)atof(it->substr(feat_pos+1).c_str());
				samp_feat.id_vec.push_back(feat_id);
				samp_feat.value_vec.push_back(feat_value);
			}
        }
        samp_feat_vec.push_back(samp_feat);
    }
    fin.close();
}


void LR::load_training_file(string training_file)
{
	cout << "Loading training data..." << endl;
	read_samp_file(training_file, samp_feat_vec, samp_class_vec);

	feat_set_size=0;
	class_set_size=0;
	for (size_t i = 0; i < samp_class_vec.size(); i++) 
	{
		if (samp_class_vec[i] > class_set_size) 
		{
			class_set_size = samp_class_vec[i];
		}
		if (samp_feat_vec[i].id_vec.back() > feat_set_size) 
		{
			feat_set_size = samp_feat_vec[i].id_vec.back();
		}	
	}
	class_set_size += 1;
	feat_set_size += 1;
}

void LR::init_omega()
{
	float init_value = 0.0;
	//float init_value = (float)1/class_set_size;
	for (int j = 0; j < class_set_size - 1; j++) {
		vector<double> temp_vec(feat_set_size, init_value);
		omega.push_back(temp_vec);
	}
}


lbfgsfloatval_t LR::evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{

	double loss=0.0;
	float acc=0.0;	

	for(int j=0;j< class_set_size - 1; j++)
		for (int i=0;i<feat_set_size;i++)		
			omega[j][i]=x[j*feat_set_size+i];

    p->calc_loss(&loss, &acc);
    lbfgsfloatval_t fx = loss;


	vector< vector<float> > delta;
	for ( j = 0; j < class_set_size - 1; j++) {
		vector<float> temp_vec(feat_set_size, 0.0);
		delta.push_back(temp_vec);
	}
	for (int k = 0; k < p->samp_class_vec.size(); k++) 
	{
		sparse_feat samp_feat = p->samp_feat_vec[k];
		int samp_class =p->samp_class_vec[k];
		vector<double> linear_sum_vec = p->calc_linear_sum(samp_feat);
		vector<double> softmax_prb_vec = p->calc_softmax_prb(linear_sum_vec);
		for (int j = 0; j < class_set_size - 1; j++)
		{
			float error = softmax_prb_vec[j] - (int)(j==samp_class);
			for (int i = 0; i < samp_feat.id_vec.size(); i++)
			{
				int feat_id = samp_feat.id_vec[i];
				float feat = samp_feat.value_vec[i];
				delta[j][feat_id] += error * feat;
			}
		}
	}

	for ( j = 0; j < class_set_size - 1; j++)
		for (int i=0;i<feat_set_size;i++)		
			g[j*feat_set_size+i]=delta[j][i];

    return fx;
}

int LR::progress(
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
    )
{
	double loss=0.0;
	float acc=0.0;	
    p->calc_loss(&loss, &acc);
	cout.setf(ios::left);
	cout << "Iter: " << setw(8) << k  << "Loss: " << setw(14) << fx << "Acc: " << setw(8) << acc << endl;
    return 0;
}

//LBFGS Optimization
int LR::train_lbfgs()
{
	int i, ret = 0;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x= lbfgs_malloc(feat_set_size*(class_set_size - 1));
	lbfgs_parameter_t param;

	if (x == NULL) {
		printf("ERROR: Failed to allocate a memory block for variables.\n");
		return 1;
	}

	/* Initialize the variables. */
	for(int j=0;j< class_set_size - 1; j++)
		for (i=0;i<feat_set_size;i++)
			x[j*feat_set_size+i]=0.0;
			
			

	/* Initialize the parameters for the L-BFGS optimization. */
	lbfgs_parameter_init(&param);
	/*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

	/*
		Start the L-BFGS optimization; this will invoke the callback functions
		evaluate() and progress() when necessary.
	*/
	ret = lbfgs(feat_set_size*(class_set_size - 1), x, &fx, evaluate, progress, NULL, &param);

	/* Report the result. */
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
	printf("loss = %f\n", fx);

	lbfgs_free(x);
	return 0;	
}

// Gradient Descent (GD) Optimization
int LR::train_gd(int max_loop, double loss_thrd, float learn_rate, float lambda)
{
	int loop = 0;
	double loss = 0.0;
	double loss_pre = 0.0;
	float acc = 0.0;

	while (loop <= max_loop) 
	{
		loss = 0.0;
		acc = 0.0;
		calc_loss(&loss, &acc);	
		cout.setf(ios::left);
		cout << "Iter: " << setw(8) << loop << "Loss: " << setw(14) << loss << "Acc: " << setw(8) << acc << endl;
		if ((loss_pre - loss) < loss_thrd && loss_pre >= loss && loop != 0)
		{
			cout << "Reaching the minimal loss decrease!" << endl;
			break;
		}
		loss_pre = loss;
		// calculate delta for each training dataset
		vector< vector<float> > delta;
		for (int j = 0; j < class_set_size - 1; j++) {
			vector<float> temp_vec(feat_set_size, 0.0);
			delta.push_back(temp_vec);
		}
		for (int k = 0; k < samp_class_vec.size(); k++) 
		{
			sparse_feat samp_feat = samp_feat_vec[k];
			int samp_class = samp_class_vec[k];
			vector<double> linear_sum_vec = calc_linear_sum(samp_feat);
			vector<double> softmax_prb_vec = calc_softmax_prb(linear_sum_vec);
			for (int j = 0; j < class_set_size - 1; j++)
			{
				float error = softmax_prb_vec[j] - (int)(j==samp_class);
				for (int i = 0; i < samp_feat.id_vec.size(); i++)
				{
					int feat_id = samp_feat.id_vec[i];
					float feat = samp_feat.value_vec[i];
					delta[j][feat_id] += error * feat;
				}
			}
		}
		// batch update omega
		for ( j = 0; j < class_set_size - 1; j++)
			for (int i = 0; i < feat_set_size; i++)
				omega[j][i] -= learn_rate * (delta[j][i] + lambda * omega[j][i]);
		
		loop++;
	}
	return 1;
}

// Stochastic Gradient Descent (SGD) Optimization
int LR::train_sgd(int max_loop, double loss_thrd, float learn_rate, float lambda) 
{
	int id = 0;
	double loss = 0.0;
	double loss_pre = 0.0;
	float acc = 0.0;
	while (id <= max_loop * samp_class_vec.size())
	{
		if (id % samp_class_vec.size() == 0) // Check after each loop of the training data
		{
			int loop = id / samp_class_vec.size();
		    loss = 0.0;
		    acc = 0.0;
			calc_loss(&loss, &acc);
			cout.setf(ios::left);
			cout << "Iter: " << setw(8) << loop << "Loss: " << setw(14) << loss << "Acc: " << setw(8) << acc << endl;
			if ((loss_pre - loss) < loss_thrd && loss_pre >= loss && id != 0)
			{
				cout << "Reaching the minimal loss decrease!" << endl;
				break;
			}
			loss_pre = loss;
		}
		// update omega for each (random) training sample
		//int samp_id = (int)(rand() % samp_class_vec.size());
		int samp_id = (int)(id % samp_class_vec.size());
		sparse_feat samp_feat = samp_feat_vec[samp_id];
		int samp_class = samp_class_vec[samp_id];
		vector<double> linear_sum_vec = calc_linear_sum(samp_feat);
		vector<double> softmax_prb_vec = calc_softmax_prb(linear_sum_vec);
		for (int j = 0; j < class_set_size - 1; j++)
		{
			float error = softmax_prb_vec[j] - (int)(j==samp_class);
			if (lambda != 0.0) 
			{
				for (int i2 = 0; i2 < feat_set_size; i2++)
				{
					omega[j][i2] -= learn_rate * lambda * omega[j][i2];
				}			
			}
			for (int i = 0; i < samp_feat.id_vec.size(); i++)
			{
				int feat_id = samp_feat.id_vec[i];
				float feat = samp_feat.value_vec[i];
				float delt = error * feat;
				omega[j][feat_id] -= learn_rate * delt;
			}
		}
		id++;
	}
	return 1;
}


void LR::calc_loss(double *loss, float *acc)
{
	double neg_log_likeli = 0.0;
	int err_num = 0;
	for (size_t k = 0; k < samp_class_vec.size(); k++) 
	{
		int samp_class = samp_class_vec[k];
		sparse_feat samp_feat = samp_feat_vec[k];
		vector<double> linear_sum_vec = calc_linear_sum(samp_feat);
		int pred_class = score_to_class(linear_sum_vec);
		if (pred_class != samp_class) 
			err_num += 1;
		vector<double> softmax_prb_vec = calc_softmax_prb(linear_sum_vec);
		for (int j = 0; j < class_set_size; j++)
	    {
			if (j == samp_class) 
			{
				double pj = softmax_prb_vec[j];
				double temp = pj < LOG_LIM ? LOG_LIM : pj;
				neg_log_likeli += log(temp);
			}
		}
	}
	*loss = -neg_log_likeli ; // Loss equals negative log-likelihood
	*acc = 1 - (float)err_num / samp_class_vec.size();
}


vector<double> LR::calc_linear_sum(sparse_feat &samp_feat)
{
	vector<double> linear_sum_vec(class_set_size, 0.0);
	for (int j = 0; j < class_set_size - 1; j++)
	{
		for (size_t i = 0; i < samp_feat.id_vec.size(); i++) 
		{
			int feat_id = samp_feat.id_vec[i];
			float feat_value = samp_feat.value_vec[i];
			linear_sum_vec[j] += omega[j][feat_id] * feat_value;
		}
	}
    return linear_sum_vec;
}


vector<double> LR::calc_softmax_prb(vector<double> &linear_sum_vec)
{
	vector<double> softmax_prb_vec(class_set_size, 0.0);
	double max_score = *(max_element(linear_sum_vec.begin(), linear_sum_vec.end()));
	double exp_sum = 0.0;
    for (int j = 0; j < class_set_size; j++) 
	{
    	softmax_prb_vec[j] = (double)exp(linear_sum_vec[j]- max_score);
    	exp_sum += softmax_prb_vec[j];
    }
	for (int j2 = 0; j2 < class_set_size; j2++)
		softmax_prb_vec[j2] /= exp_sum;
	return softmax_prb_vec;
}


int LR::score_to_class(vector<double> &score_vec)
{
	int pred_class = 0;	
	double max_score = score_vec[0];
	for (int j = 1; j < class_set_size; j++) {
		if (score_vec[j] > max_score) {
			max_score = score_vec[j];
			pred_class = j;
		}
	}
    return pred_class;
}

float LR::classify_testing_file(string testing_file, string output_file, int output_format)
{
	cout << "Classifying testing file..." << endl;
	vector<sparse_feat> test_feat_vec;
	vector<int> test_class_vec;
	vector<int> pred_class_vec;
	read_samp_file(testing_file, test_feat_vec, test_class_vec);
	ofstream fout(output_file.c_str());
	for (size_t i = 0; i < test_class_vec.size(); i++) 
	{
		int samp_class = test_class_vec[i];
		sparse_feat samp_feat = test_feat_vec[i];
		vector<double> pred_score = calc_linear_sum(samp_feat);			
		int pred_class = score_to_class(pred_score);
		pred_class_vec.push_back(pred_class);
		fout << pred_class << "\t";
		if (output_format == 1) 
		{
			for (int j = 0; j < class_set_size; j++) 
			{
				fout << pred_score[j] << ' '; 
			}		
		}
		else if (output_format == 2) 
		{
			vector<double> pred_prb = calc_softmax_prb(pred_score);
			for (int j = 0; j < class_set_size; j++)
			{
				fout << pred_prb[j] << ' '; 
			}
		}

		fout << endl;		
	}
	fout.close();
	float acc = calc_acc(test_class_vec, pred_class_vec);
	return acc;
}

float LR::calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec)
{
	size_t len = test_class_vec.size();
	if (len != pred_class_vec.size()) {
		cerr << "Error: two vectors should have the same lenght." << endl;
		exit(0);
	}
	int err_num = 0;
	for (size_t id = 0; id != len; id++) {
		if (test_class_vec[id] != pred_class_vec[id]) {
			err_num++;
		}
	}
	return 1 - ((float)err_num) / len;
}


vector<string> LR::string_split(string terms_str, string spliting_tag)
{
	vector<string> feat_vec;
    size_t term_beg_pos = 0;
    size_t term_end_pos = 0;
    while ((term_end_pos = terms_str.find_first_of(spliting_tag, term_beg_pos)) != string::npos) 
	{
        if (term_end_pos > term_beg_pos)
		{
            string term_str = terms_str.substr(term_beg_pos, term_end_pos - term_beg_pos);
            feat_vec.push_back(term_str);
        }
        term_beg_pos = term_end_pos + 1;
    }
	if (term_beg_pos < terms_str.size())
	{
		string end_str = terms_str.substr(term_beg_pos);
		feat_vec.push_back(end_str);
	}
    return feat_vec;
}






