#include <iostream>
#include <algorithm>
#include <math.h>

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
	//meshRefinement.uniformly_refine(3);
	
	//1D - for debugging
	//int n = 200;
	//MeshTools::Generation::build_line(mesh, n, 0.0, 1.0, EDGE2); //n linear elements from 0 to 1
	
	//nice geometry for debugging
	MeshTools::Generation::build_square (mesh, 
																					120, 40,
                                         -1.0, 2.0,
                                         -1.0, 1.0,
                                         QUAD9);

	MeshBase::element_iterator       elem_it  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end = mesh.elements_end();
  for (; elem_it != elem_end; ++elem_it){
    Elem* elem = *elem_it;
    Point c = elem->centroid();
    //if(c.size()<0.8)
    //if(c(0)<0)
        elem->subdomain_id() = 0;
    //else
    		//elem->subdomain_id() = 2;
  }


	EquationSystems equation_systems (mesh);

	#ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems("meep.exo",equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

	return 0;
} // end main
