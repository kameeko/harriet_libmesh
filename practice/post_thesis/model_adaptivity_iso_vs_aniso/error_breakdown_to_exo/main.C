// C++ includes
#include <iomanip>

// Basic include files
#include "libmesh/equation_systems.h"
#include "libmesh/error_vector.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/uniform_refinement_estimator.h"
#include "libmesh/getpot.h"
#include "libmesh/enum_xdr_mode.h" //DEBUG
#include "libmesh/gmv_io.h" //DEBUG
#include "libmesh/direct_solution_transfer.h"

// The systems and solvers we may use
#include "libmesh/diff_solver.h"
#include "libmesh/steady_solver.h"
#include "libmesh/newton_solver.h"
#include "dummy_sys.h"

int main(int argc, char** argv){
//convert error breakdown from psi_and_superadj runs to exo file that paraview can display

  // Initialize libMesh
  LibMeshInit init(argc, argv);

  Mesh mesh(init.comm());
  GetPot infile("contamTrans.in");
  std::string find_mesh_here = infile("divided_mesh","divvy1.exo");
	mesh.read(find_mesh_here);
	
	// Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  
  DummySys & dummy_sys = equation_systems.add_system<DummySys>("Blah");
  
  dummy_sys.time_solver = AutoPtr<TimeSolver>(new SteadySolver(dummy_sys));
  
  // Initialize the system
	equation_systems.init ();
	
	//"solve" dummy system
	dummy_sys.solve();
	
	//read in error breakdown
	std::string find_error_here = infile("error_file","error_est_breakdown_basis_blame0.dat");
	std::vector<Real> err_vec;
  if(FILE *fp=fopen(find_error_here.c_str(),"r")){
		Real eep;
		int flag = 1;
		while(flag != -1){
			flag = fscanf(fp, "%lf",&eep);
			if(flag != -1){
				err_vec.push_back(eep);
			}
		}
	}
	
	//consolidate error breakdown into nodal contributions
	std::vector<Real> err_vec_squish;
	//AutoPtr<NumericVector<Number> > errNumVec = dummy_sys.solution->zero_clone();
	//std::cout << errNumVec->size() << " " << round(err_vec.size()/6.) << std::endl; //check
	for(unsigned int node_num = 0; node_num < round(err_vec.size()/6.); node_num++){
	  err_vec_squish.push_back(fabs(err_vec[node_num*6] 
                                 + err_vec[node_num*6+1] 
                                 + err_vec[node_num*6+2]
                                 + err_vec[node_num*6+3] 
                                 + err_vec[node_num*6+4] 
                                 + err_vec[node_num*6+5]));
	}
	//*errNumVec = err_vec_squish;
	
	//transfer consolidated error breakdown into "solution"
	const std::string & adjoint_solution0_name = "adjoint_solution0";
  dummy_sys.add_vector(adjoint_solution0_name, false, GHOSTED);
  dummy_sys.set_vector_as_adjoint(adjoint_solution0_name,0);
  dummy_sys.set_adjoint_already_solved(true);
  NumericVector<Number> &dual_sol = dummy_sys.get_adjoint_solution(0);
  dual_sol = err_vec_squish;
  NumericVector<Number> &primal_sol = *dummy_sys.solution;
  dual_sol.swap(primal_sol);
	ExodusII_IO(mesh).write_timestep("err_breakdown.exo", equation_systems, 1, dummy_sys.time);

// All done.
  return 0;

}
