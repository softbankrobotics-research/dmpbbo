
/**
 * \file demoImitationAndOptimization.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to initialize a DMP, and then optimize it with an evolution strategy.
 *
 * \ingroup Demos
 *
 * This file is part of DmpBbo, a set of libraries and programs for the
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
 *
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <python2.7/Python.h>
#include <string>
#include <set>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <vector>
#include "dmpbbo_io/EigenFileIO.hpp"


#include "dmp_bbo/tasks/TaskViapoint.hpp"
#include "dmp_bbo/TaskWithTrajectoryDemonstrator.hpp"
#include "dmp_bbo/TaskSolverDmp.hpp"

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"

#include "functionapproximators/FunctionApproximatorLWPR.hpp"
#include "functionapproximators/MetaParametersLWPR.hpp"

#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"


#include "functionapproximators/demos/targetFunction.hpp"

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/updaters/UpdaterCovarDecay.hpp"
#include "bbo/updaters/UpdaterCovarAdaptation.hpp"



#include <iostream>


using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Run a learning session in which a Dmp is first trained with a trajectory, and then optimized with an evolution strategy w.r.t. a cost function.
 * \param[in] function_approximators The function approximators for the Dmp (a vector; one element for each Dmp dimension)
 * \param[in] trajectory The trajectory with which to train the Dmp
 * \param[in] task The task to optimize (it contains the cost function)
 * \param[in] updater The parameter updater to use during optimization
 * \param[in] directory Directory to which to save results
 */




int main(int n_args, char* args[])
{

  if (n_args != 3)
  {
      cerr << "Usage : <path_to_trajectory> <nb_basis_functions>" << endl;
      exit(1);
  }

  //****
  //****
  // READING TRAJECTORY
  //****
  //****



  // this is the trajectory that we want to TRAIN the DMP on
  int n_basis_functions;
  istringstream (args[2]) >> n_basis_functions;

  string trajectory_file = args[1];
  cout << "Reading trajectory from TXT file: " << trajectory_file << endl;
  Trajectory traj_demo = Trajectory::readFromFile(trajectory_file);

  double tau = traj_demo.duration();
  int n_time_steps = traj_demo.length();
  VectorXd ts = traj_demo.ts(); // Time steps
  int n_dims_dmp = traj_demo.dim();




  //****
  //****
  // FUNCTION APPROXIMATORS
  //****
  //****

  //MatrixXd traj_matrix;
  //loadMatrix(trajectory_file, traj_matrix);
  //int n_time_steps = traj_matrix.row(0).size();
  //MatrixXd inputs = VectorXd::LinSpaced(n_time_steps,0.0,1.0);
  //MatrixXd targets = traj_matrix.block(0, 1, n_time_steps,  3*n_dims_dmp-1);

  // Initialize some meta parameters for training LWR function approximator
  int input_dim = 1;
  double intersection = 0.5;
  //MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions,intersection);
  //FunctionApproximatorLWR* fa_ptr = new FunctionApproximatorLWR(meta_parameters);
  //int number_of_gaussians = pow(5,input_dim);
  //MetaParametersGMR* meta_parameters = new MetaParametersGMR(input_dim,n_basis_functions);
  //FunctionApproximatorGMR* fa_ptr = new FunctionApproximatorGMR(meta_parameters);
  //fa_ptr->train(inputs,targets);

  // RBFN 
  MetaParametersRBFN *meta_parameters_rbfn= new MetaParametersRBFN(input_dim,n_basis_functions);
  FunctionApproximatorRBFN* fa_ptr = new FunctionApproximatorRBFN(meta_parameters_rbfn);


  //double   w_gen=0.2;
  //double   w_prune=0.8;
  //bool     update_D=true;
  //double   init_alpha=0.1;
  //double   penalty=0.005;
  //VectorXd init_D=VectorXd::Constant(input_dim,n_basis_functions);
  //MetaParametersLWPR* meta_parameters_lwpr = new MetaParametersLWPR(input_dim,init_D,w_gen,w_prune,update_D,init_alpha,penalty);
  //FunctionApproximatorLWPR* fa_ptr = new FunctionApproximatorLWPR(meta_parameters_lwpr);

  // Clone the function approximator for each dimension of the DMP
  vector<FunctionApproximator*> function_approximators(n_dims_dmp);
  for (int dd=0; dd<n_dims_dmp; dd++)
    function_approximators[dd] = fa_ptr->clone();




  //****
  //****
  // DMP CREATION AND TRAINING
  //****
  //****


  Trajectory traj_reproduced; // this is where the new DMP trajectory will be saved in

  Dmp* dmp = new Dmp(n_dims_dmp, function_approximators, Dmp::KULVICIUS_2012_JOINING);
  dmp->train(traj_demo);
  dmp->analyticalSolution(traj_demo.ts(),traj_reproduced);






  // Make the task solver
  set<string> parameters_to_optimize;
  //parameters_to_optimize.insert("priors");
  //parameters_to_optimize.insert("slopes");
  //parameters_to_optimize.insert("centers");
  //parameters_to_optimize.insert("weights");
  //parameters_to_optimize.insert("centers");
  //parameters_to_optimize.insert("widths");
  //parameters_to_optimize.insert("offsets");

  //LWR
  //parameters_to_optimize.insert("centers");
  //parameters_to_optimize.insert("widths");
  //parameters_to_optimize.insert("offsets");
  //parameters_to_optimize.insert("slopes");

  //RBFN
  parameters_to_optimize.insert("centers");
  // parameters_to_optimize.insert("widths");
  // parameters_to_optimize.insert("weights");


  // put the fact that I need to optimize the parameters
  dmp->setSelectedParameters(parameters_to_optimize);


  // Save the new trajectory in a file
  bool overwrite=true;
  string save_directory_trained= "tmp/demoDmpOptimize";
  traj_reproduced.saveToFile(save_directory_trained,"trained_trajectory.txt",overwrite);



  //****
  //****
  // CREATION OF A DISTRIBUTION AROUND THE DMP PARAMETERS
  //****
  //****

  // Make the initial distribution
  cout << "getParam init" << endl;
  vector<VectorXd> mean_init_vec;
  dmp->getParameterVectorSelected(mean_init_vec);

  cout << "mean_init_vect : " << endl;
  for (int i = 0 ; i< mean_init_vec.size(); i++)
  {
      cout << endl << "dimension " << i+1 << " : " << endl  << endl << mean_init_vec[i] << endl;
  }

  ofstream myfile;
  myfile.open("tmp/parameters.txt",fstream::in | fstream::out | fstream::trunc);
  if (myfile.is_open())
  {
    for (int i = 0 ; i< mean_init_vec.size(); i++)
    {
        for (int j=0; j<mean_init_vec[i].size(); j++)
        {
            myfile << mean_init_vec[i][j]<< " ";
        }
        myfile << endl;

    }
    myfile.close();
  }
  else cout << "Unable to open file" << endl;

}
