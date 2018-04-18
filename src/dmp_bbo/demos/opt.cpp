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

#include <string>
#include <set>
#include <fstream>
#include <eigen3/Eigen/Core>

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
#include "functionapproximators/ModelParametersGMR.hpp"

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

#include "bbo/DistributionGaussian.hpp"
#include "bbo/Updater.hpp"
#include "bbo/updaters/UpdaterCovarDecay.hpp"
#include "bbo/updaters/UpdaterCovarAdaptation.hpp"






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



void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}


void readFromFile(string filename, MatrixXd& samples)
{
  //MatrixXd samples;
  if (!loadMatrix(filename,samples))
  {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Cannot open filename '"<< filename <<"'." << endl;
    return;
  }

  int n_dims       = samples.rows()-1;
  int n_parameters = samples.cols();

  //removeColumn(samples, n_parameters-1); // remove the last column because for some reason it repeats the last value twice

}




// obtains a trajectory from the sample provided
void obtainTrajectory(const MatrixXd& samples, string save_directory, Dmp *dmp, int n_time_steps, VectorXd ts)
{
  int n_dofs = samples.rows();
  vector<VectorXd> model_parameters_vec(n_dofs);
  for (int dd=0; dd<n_dofs; dd++)
      model_parameters_vec[dd] = samples.row(dd);

  dmp->setParameterVectorSelected(model_parameters_vec, false);

  int n_dims = dmp->dim(); // Dimensionality of the system
  MatrixXd xs_ana(n_time_steps,n_dims);
  MatrixXd xds_ana(n_time_steps,n_dims);
  MatrixXd forcing_terms(n_time_steps,n_dofs);
  forcing_terms.fill(0.0);
  dmp->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms);

  Trajectory traj_reproduced;
  dmp->analyticalSolution(ts,traj_reproduced);
  traj_reproduced.saveToFile(save_directory,"traj.txt",true);


}






int main(int n_args, char* args[])
{

  MatrixXd samples;
  //string initial_trajectory_file="/home/asya/Desktop/DMP_Applications/Pepper_Stuff/OPTIMIZATION/throw.txt";
  if (n_args != 4)
  {
      cerr << "Usage : <path_to_trajectory> <path_to_sample_file> <nb_basis_functions>" << endl;
      exit(1);
  }
  // this is the trajectory that we want to TRAIN the DMP on
  string initial_trajectory_file = args[1];
  string sample_file = args[2];
  Trajectory traj_demo = Trajectory::readFromFile(initial_trajectory_file);

  int n_basis_functions;
  istringstream (args[3]) >> n_basis_functions;

  double tau = traj_demo.duration();
  int n_time_steps = traj_demo.length();
  VectorXd ts = traj_demo.ts(); // Time steps
  int n_dims_dmp = traj_demo.dim();




  //****
  //****
  // FUNCTION APPROXIMATORS
  //****
  //****




  // Initialize some meta parameters for training LWR function approximator
  int input_dim = 1;
  double intersection = 0.5;
  //MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions,intersection);
  //FunctionApproximatorLWR* fa_ptr = new FunctionApproximatorLWR(meta_parameters);
  int number_of_gaussians = pow(5,input_dim);
  //MetaParametersGMR* meta_parameters = new MetaParametersGMR(input_dim,n_basis_functions);
  //FunctionApproximatorGMR* fa_ptr = new FunctionApproximatorGMR(meta_parameters);

  //MetaParametersRBFN *meta_parameters_rbfn= new MetaParametersRBFN(input_dim,n_basis_functions);
  //FunctionApproximatorRBFN* fa_ptr = new FunctionApproximatorRBFN(meta_parameters_rbfn);

  double   w_gen=0.2;
  double   w_prune=0.8;
  bool     update_D=true;
  double   init_alpha=0.1;
  double   penalty=0.005;
  VectorXd init_D=VectorXd::Constant(input_dim,n_basis_functions);
  MetaParametersLWPR* meta_parameters_lwpr = new MetaParametersLWPR(input_dim,init_D,w_gen,w_prune,update_D,init_alpha,penalty);
  FunctionApproximatorLWPR* fa_ptr = new FunctionApproximatorLWPR(meta_parameters_lwpr);

  // Clone the function approximator for each dimension of the DMP
  vector<FunctionApproximator*> function_approximators(n_dims_dmp);
  for (int dd=0; dd<n_dims_dmp; dd++)
    function_approximators[dd] = fa_ptr->clone();




  // Initialize and train DMP

  Dmp* dmp = new Dmp(n_dims_dmp, function_approximators, Dmp::KULVICIUS_2012_JOINING);

  dmp->train(traj_demo);
  //set<string> parameters_to_optimize;
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
  //parameters_to_optimize.insert("centers");
  //parameters_to_optimize.insert("widths");
  //parameters_to_optimize.insert("weights");
  //dmp->setSelectedParameters(parameters_to_optimize);


  readFromFile(sample_file, samples);
  obtainTrajectory(samples, "tmp/updatedTrajs", dmp, n_time_steps, ts);



  return 0;

}


