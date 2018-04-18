/**
 * \file demoDmp.cpp
 * \author Freek Stulp
 * \brief  Demonstrates how to initialize, train and integrate a Dmp.
 *
 * \ingroup Demos
 * \ingroup Dmps
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

#include "dmp/Dmp.hpp"
#include "dmp/Trajectory.hpp"

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "dynamicalsystems/ExponentialSystem.hpp"
#include "dynamicalsystems/SigmoidSystem.hpp"
#include "dynamicalsystems/TimeSystem.hpp"
#include "dynamicalsystems/SpringDamperSystem.hpp"

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/MetaParametersLWR.hpp"
#include "functionapproximators/ModelParametersLWR.hpp"

#include "functionapproximators/FunctionApproximatorRBFN.hpp"
#include "functionapproximators/MetaParametersRBFN.hpp"
#include "functionapproximators/ModelParametersRBFN.hpp"

#include "functionapproximators/FunctionApproximatorGMR.hpp"
#include "functionapproximators/MetaParametersGMR.hpp"
#include "functionapproximators/ModelParametersGMR.hpp"



#include "dmpbbo_io/EigenFileIO.hpp"

#include <iostream>
#include <fstream>



using namespace std;
using namespace Eigen;
using namespace DmpBbo;

/** Get a demonstration trajectory.
 * \param[in] ts The time steps at which to sample
 * \return a Demonstration trajectory
 */
Trajectory getDemoTrajectory(const VectorXd& ts);

/** Main function
 * \param[in] n_args Number of arguments
 * \param[in] args Arguments themselves
 * \return Success of exection. 0 if successful.
 */
int main(int n_args, char** args)
{
  /*
  ******** if n=3 : NO NEW END POINT AND NO NEW START POINT
  ******** if n=9 : NEW END POINT BUT NO NEW START POINT
  ******** if n=10 : NEW START POINT BUT NO NEW END POINT (here, I will just send an arbitrary argument at the end)
  ******** if n=15 : NEW END POINT AND NEW START POINT
  */

  // if n=3 then I will not pick a new goal point
  if (n_args!=4)
  {
    cerr << "Usage: " << args[0] << "<number_basis_functions> <trajectory_file> <save_directory> " << endl;
    return -1;
  }
  int n_basis_functions;
  istringstream (args[1]) >> n_basis_functions;
  string input_robot_trajectory = string(args[2]); //="/home/asya/Desktop/DMP_Applications/Letter-Drawing/Pepper_Stuff/output.txt";
  string save_directory=string(args[3]);
  //string output_xml_file("/tmp/dmp.xml");






  cout << "Reading trajectory from TXT file: " << input_robot_trajectory << endl;
  Trajectory trajectory = Trajectory::readFromFile(input_robot_trajectory);

  double tau = trajectory.duration();
  int n_time_steps = trajectory.length();
  VectorXd ts = trajectory.ts(); // Time steps
  int n_dims = trajectory.dim();

  cout << "n_dims is : " << n_dims;



  // MAKE THE FUNCTION APPROXIMATORS




  // LWR
  //int n_basis_functions = 20;
  int input_dim = 1;
  double intersection = 0.5;
  MetaParametersLWR* meta_parameters = new MetaParametersLWR(input_dim,n_basis_functions,intersection);
  FunctionApproximatorLWR* fa_lwr = new FunctionApproximatorLWR(meta_parameters);

  // Clone the function approximator for each dimension of the DMP
  vector<FunctionApproximator*> function_approximators(n_dims);
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = fa_lwr->clone();


/*

  //RBFN
  int input_dim=1;
  int n_basis_functions=90;
  MetaParametersRBFN* meta_parameters=new MetaParametersRBFN(input_dim, n_basis_functions, 0.5);
  FunctionApproximatorRBFN* fa_rbfn=new FunctionApproximatorRBFN(meta_parameters);

  vector<FunctionApproximator*> function_approximators(n_dims);
  for (int dd=0; dd<n_dims; dd++)
    function_approximators[dd] = fa_rbfn->clone();


*/

  // CONSTRUCT AND TRAIN THE DMP

  // Initialize the DMP
  Dmp* dmp = new Dmp(n_dims, function_approximators, Dmp::KULVICIUS_2012_JOINING);

  // And train it. Passing the save_directory will make sure the results are saved to file.
  bool overwrite = true;
  dmp->train(trajectory,save_directory,overwrite);



  /*
  float new_duration=tau;
  float ratio=tau/(new_duration);
  dmp->set_tau(new_duration);
  if (ratio>1) n_time_steps=n_time_steps/(2*ratio); // if the new duration is smaller than I want to create less time steps : allows the drawing to work properly (lags if too many time steps)
  */


  // INTEGRATE DMP TO GET REPRODUCED TRAJECTORY
  Trajectory traj_reproduced;
  ts = VectorXd::LinSpaced(n_time_steps,0,tau); // Time steps
  dmp->analyticalSolution(ts,traj_reproduced);

  // Integrate again, but this time get more information
  MatrixXd xs_ana, xds_ana, forcing_terms_ana, fa_output_ana;
  dmp->analyticalSolution(ts,xs_ana,xds_ana,forcing_terms_ana,fa_output_ana);


  // WRITE THINGS TO FILE
  cout << "Saving newly DMP trajectories to: " << save_directory << endl;
  trajectory.saveToFile(save_directory,"demonstration_traj.txt",overwrite);
  traj_reproduced.saveToFile(save_directory,"reproduced_traj.txt",overwrite);

  MatrixXd output_ana(ts.size(),1+xs_ana.cols()+xds_ana.cols());
  output_ana << xs_ana, xds_ana, ts;
  /*
  saveMatrix(save_directory,"reproduced_xs_xds.txt",output_ana,overwrite);
  saveMatrix(save_directory,"reproduced_forcing_terms.txt",forcing_terms_ana,overwrite);
  saveMatrix(save_directory,"reproduced_fa_output.txt",fa_output_ana,overwrite);
  */

  delete meta_parameters;
  delete fa_lwr;
  //delete fa_rbfn;
  //delete fa_gmr;
  delete dmp;

  return 0;
}
