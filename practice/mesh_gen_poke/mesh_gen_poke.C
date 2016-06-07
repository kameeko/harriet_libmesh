#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <fstream>

#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/vtk_io.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/equation_systems.h"
#include "libmesh/mesh_refinement.h"

#include "libmesh/fe.h"

#include "libmesh/quadrature_gauss.h"

#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"

#include "libmesh/dof_map.h"

#include "libmesh/exodusII_io.h"

using namespace libMesh;

int main (int argc, char** argv){

	LibMeshInit init (argc, argv); //initialize libmesh library

	std::cout << "Running " << argv[0];
	for (int i=1; i<argc; i++)
		std::cout << " " << argv[i];
	std::cout << std::endl << std::endl;

	Mesh mesh(init.comm());
	
	//T-channel
	//mesh.read("mesh.e");
	//MeshRefinement meshRefinement(mesh);
	//meshRefinement.uniformly_refine(0);
	
	//1D - for debugging
	//int n = 20;
	//MeshTools::Generation::build_line(mesh, n, 0.0, 1.0, EDGE2); //n linear elements from 0 to 1
	
	//nice geometry (straight channel) 
	MeshTools::Generation::build_square (mesh, 
																					250, 50,
                                         -0.0, 5.0,
                                         -0.0, 1.0,
                                         QUAD9);

	//read in subdomain assignments
	
	std::vector<double> prev_assign(mesh.n_elem(), 0.);
	std::string read_assign = "do_divvy.txt";
	if(FILE *fp=fopen(read_assign.c_str(),"r")){
		int flag = 1;
		int elemNum, assign;
		int ind = 0;
		while(flag != -1){
			flag = fscanf(fp, "%d %d",&elemNum,&assign);
			if(flag != -1){
				prev_assign[ind] = assign;
				ind += 1;
			}
		}
		fclose(fp);
	}
	

	//to stash subdomain assignments
	std::string stash_assign = "divvy.txt";
	std::ofstream output(stash_assign.c_str());

	MeshBase::element_iterator       elem_it  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end = mesh.elements_end();
  for (; elem_it != elem_end; ++elem_it){
    Elem* elem = *elem_it;
    Point c = elem->centroid();
    //Point cshift1(c(0)-0.63, c(1)-0.5);

    elem->subdomain_id() = prev_assign[elem->id()];
    //TEST/DEBUG
    //if(fabs(c(0)-3.0) < 0.35 && fabs(c(1)-0.5) < 0.35)
    //  elem->subdomain_id() = 1;
    		
    if(output.is_open()){
    	output << elem->id() << " " << elem->subdomain_id() << "\n";
    }
  }
  output.close();


	EquationSystems equation_systems (mesh);

	#ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems("meep.exo",equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

	return 0;
} // end main
