/********************************************************************
* Logit Model (Softmax Regression) V0.20
* Implemented by Rui Xia(rxiacn@gmail.com)
* Last updated on 2013-05-04.   
*********************************************************************/


#include <cstdlib>
#include <iostream>
#include <cstring> 

#include "LR.h"
#include "lbfgs.h"



using namespace std;




void print_help() 
{
	cout << "\nOpenPR-LR training module, " << VERSION << ", " << VERSION_DATE << "\n\n"
		<< "usage: lr_train [options] training_file model_file [pre_model_file]\n\n"
		<< "options: -h         -> help\n"
		<< "         -o [0,1,2] -> 0: gradient descent (default)\n"
		<< "                    -> 1: stochastic gradient descent \n"
		<< "                    -> 2: L-BFGS \n"
		<< "         -n int     -> maximal iteration loops (default 200)\n"
		<< "         -m double  -> minimal loss value decrease (default 1e-03)\n"
		<< "         -r double  -> lambda of gaussian prior regularization (default 0)\n"		
		<< "         -l float   -> learning rate (default 1.0)\n"
		<< "         -u [0,1]   -> 0: initial training model (default)\n"
		<< "                    -> 1: updating model (pre_model_file is needed)\n" 
		<< endl;
}

void read_parameters(int argc, char *argv[], char *training_file, char *model_file, 
					 int *opt_method, int *max_loop, double *loss_thrd, float *learn_rate,
						 float *lambda, int *update, char *pre_model_file)
{
	// set default options
	*opt_method = 2;
	*max_loop = 200;
	*loss_thrd = 1e-3;
	*learn_rate = 1.0;
	*lambda = 0.0;
	*update = 0;
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) 
	{
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			case 'o':
				*opt_method = atoi(argv[++i]);
				break;
			case 'n':
				*max_loop = atoi(argv[++i]);
				break;
			case 'm':
				*loss_thrd = atof(argv[++i]);
				break;
			case 'l':
				*learn_rate = (float)atof(argv[++i]);
				break;
			case 'r':
				*lambda = (float)atof(argv[++i]);
				break;
			case 'u':
				*update = atoi(argv[++i]);
				break;
			default:
				cout << "Unrecognized option: " << argv[i] << "!" << endl;
				print_help();
				exit(0);
		}
	}
	
	if ((i+1)>=argc) 
	{
		cout << "Not enough parameters!" << endl;
		print_help();
		exit(0);
	}

	strcpy (training_file, argv[i]);
	strcpy (model_file, argv[i+1]);
	if (*update) 
	{
		if ((i+2)>=argc) 
		{
			cout << "Previous model file is needed in update mode!" << endl;
			print_help();
			exit(0);
		}
		strcpy (pre_model_file, argv[i+2]);
	}
}




int logit_train(int argc, char *argv[])
{
	char training_file[200];
	char model_file[200];
	int opt_method;
	int max_loop;
	double loss_thrd;
	float learn_rate;
	float lambda;
	int update;
	char pre_model_file[200];
	read_parameters(argc, argv, training_file, model_file, &opt_method, &max_loop, &loss_thrd, &learn_rate, &lambda, &update, pre_model_file);
    
    LR Logit;
    Logit.load_training_file(training_file);
    if (update) 
	{
		Logit.load_model(pre_model_file);
	}
	else 
	{
		Logit.init_omega();	
	}

	if (opt_method == 0)
	{
		Logit.train_gd(max_loop, loss_thrd, learn_rate, lambda);
	}
	else if(opt_method == 1)
	{
		Logit.train_sgd(max_loop, loss_thrd, learn_rate, lambda);
	}
	else
	{
		Logit.train_lbfgs();	
	}
    
	Logit.save_model(model_file);
	return 0;
}



int main(int argc, char *argv[])
{
    return logit_train(argc, argv);
}
