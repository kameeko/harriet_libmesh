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
#include "libmesh/elem.h"
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

//do elems that are not refined keep their elem ids?

int main (int argc, char** argv){

	LibMeshInit init (argc, argv); //initialize libmesh library

	Mesh mesh(init.comm());
	
	//initial mesh
	MeshTools::Generation::build_square (mesh, 
																					5, 5,
                                         -0.0, 1.0,
                                         -0.0, 1.0,
                                         QUAD9);

  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.max_h_level() = 1; //only controls how much refinement can be done in one go...

  //initial elem ids
	std::string stash_assign = "coarse.txt";
	std::ofstream output(stash_assign.c_str());
	MeshBase::element_iterator       elem_it  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end = mesh.elements_end();
  for (; elem_it != elem_end; ++elem_it){
    Elem* elem = *elem_it;
    Point c = elem->centroid();
    		
    if(output.is_open()){
    	output << elem->id() << " " << c(0) << " " << c(1) << "\n";
    }
  }
  output.close();
  
  //refine elements by id
  std::set<dof_id_type> refine_these;
  refine_these.insert(10);
  elem_it = mesh.elements_begin();
  for (; elem_it != elem_end; ++elem_it){
    Elem* elem = *elem_it;
    dof_id_type elem_id = elem->id();
    if(refine_these.find(elem_id) != refine_these.end()){
      elem->set_refinement_flag(Elem::REFINE);
    }
  }
  
  int prev_nelems = mesh.n_elem();
  mesh_refinement.refine_elements();
  mesh_refinement.refine_elements(); //does calling refine twice break things?
  //mesh_refinement.uniformly_refine(1);
  int new_nelems = mesh.n_elem();
  for(int i = 0; i < (new_nelems-prev_nelems); i++){
    mesh.elem(prev_nelems+i)->subdomain_id() = 1;
  }
  
  //test how max refinement level holds out
  refine_these.insert(12);
  //refine_these.erase(10);
  //refine_these.insert(25); //will still work even with max_h_level = 1
  mesh_refinement.clean_refinement_flags();
  MeshBase::element_iterator       elem_it3  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end3 = mesh.elements_end();
  for (; elem_it3 != elem_end3; ++elem_it3){
    Elem* elem = *elem_it3;
    dof_id_type elem_id = elem->id();
    if((refine_these.find(elem_id) != refine_these.end()) && elem->active()) {
      elem->set_refinement_flag(Elem::REFINE);
    }
  }
  mesh_refinement.refine_elements(); //does asking to refine 10 again break things?
  
  //post-refinement elem ids
	std::string stash_assign2 = "refined.txt";
	std::ofstream output2(stash_assign2.c_str());
	MeshBase::element_iterator       elem_it2  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end2 = mesh.elements_end();
  for (; elem_it2 != elem_end2; ++elem_it2){
    Elem* elem = *elem_it2;
    Point c = elem->centroid();
    		
    if(output2.is_open()){
    	output2 << elem->id() << " " << c(0) << " " << c(1) << "\n";
    }
  }
  output2.close();
//refined element(s) remain but become inactive...  
  mesh.print_info();

	EquationSystems equation_systems (mesh);

	#ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems("meep.exo",equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

	return 0;
} // end main
