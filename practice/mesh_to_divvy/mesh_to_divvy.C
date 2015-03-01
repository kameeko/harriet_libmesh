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
	
	mesh.read("channel_long.exo");
	
	std::string stash_assign = "divvy.txt";
	std::ofstream output(stash_assign.c_str());

	MeshBase::element_iterator       elem_it  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end = mesh.elements_end();
  for (; elem_it != elem_end; ++elem_it){
    Elem* elem = *elem_it;
    		
    if(output.is_open()){
    	output << elem->id() << " " << elem->subdomain_id() << "\n";
    }
  }
  output.close();
	
	return 0;
}
