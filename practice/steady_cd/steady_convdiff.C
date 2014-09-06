#include <iostream>
#include <algorithm>
#include <math.h>

#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/vtk_io.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/equation_systems.h"

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

void assemble_convdiff(EquationSystems& es,
                    const std::string& system_name);

Real exact_solution (const Real x, const Real y, const Real z = 0.0){
	return exp(-0.5*(x*x+y*y));
	//return x+y;
}

int main (int argc, char** argv){

	LibMeshInit init (argc, argv); //initialize libmesh library

	std::cout << "Running " << argv[0];
	for (int i=1; i<argc; i++)
		std::cout << " " << argv[i];
	std::cout << std::endl << std::endl;

	Mesh mesh(init.comm());

	MeshTools::Generation::build_square (mesh,
		                                   40, 40, //40x40 elements
		                                   -1., 1., //from x=-1 to x=1
		                                   -1., 1., //from y=-1 to y=1
		                                   QUAD9); //quadrilaterals with 9 DOF instead of default 4

	mesh.print_info();

	EquationSystems equation_systems (mesh);

	equation_systems.add_system<LinearImplicitSystem> ("ConvDiff");
	equation_systems.get_system("ConvDiff").add_variable("u", SECOND);
	equation_systems.get_system("ConvDiff").attach_assemble_function (assemble_convdiff);

	equation_systems.init();
	equation_systems.print_info();
	equation_systems.get_system("ConvDiff").solve();

	#if defined(LIBMESH_HAVE_VTK) && !defined(LIBMESH_ENABLE_PARMESH)
	VTKIO (mesh).write_equation_systems ("out.pvtu", equation_systems);
	#endif // #ifdef LIBMESH_HAVE_VTK
	
	#ifdef LIBMESH_HAVE_EXODUS_API
    ExodusII_IO (mesh).write_equation_systems("concentration.exo",equation_systems);
  #endif // #ifdef LIBMESH_HAVE_EXODUS_API

	return 0;
} // end main



void assemble_convdiff(EquationSystems& es, const std::string& system_name){

	//make sure it's attached to right system
	libmesh_assert_equal_to (system_name, "ConvDiff"); 

	const MeshBase& mesh = es.get_mesh(); //get reference to mesh
	const unsigned int dim = mesh.mesh_dimension(); //dimension of mesh

	//get reference to system being solved
	LinearImplicitSystem& system = es.get_system<LinearImplicitSystem> ("ConvDiff");

	//get reference to DofMap that handles index translation from element and node 
	//number to global DoF number
	const DofMap& dof_map = system.get_dof_map(); 
	
	//get reference to FE type of first variable in system
	FEType fe_type = dof_map.variable_type(0);

	AutoPtr<FEBase> fe (FEBase::build(dim, fe_type));
	QGauss qrule (dim, FIFTH); //5th order Gauss quadrature, for element integration
	fe->attach_quadrature_rule (&qrule);

	AutoPtr<FEBase> fe_face (FEBase::build(dim, fe_type));
	QGauss qface(dim-1, FIFTH); //5th order Gauss quadrature, for boundary integration
	fe_face->attach_quadrature_rule (&qface);

	const std::vector<Real>& JxW = fe->get_JxW(); //element Jacobian quadrature weight at integration points
	const std::vector<Point>& q_point = fe->get_xyz(); //locations of quadrature points
	const std::vector<std::vector<Real> >& phi = fe->get_phi(); //element shape functions at quadrature points
	const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi(); //element shape function gradients at quad pts

	DenseMatrix<Number> Ke; //local stiffness matrix
	DenseVector<Number> Fe; //local contribution to RHS

	std::vector<dof_id_type> dof_indices; //DoF indices for each element

	//iterate over all elements
	MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end(); //stop here

	for ( ; el != end_el ; ++el){
	  const Elem* elem = *el; //store pointer to current element

	  dof_map.dof_indices (elem, dof_indices); //get DoF indices for current element
	  fe->reinit (elem); //compute element-specific data (quad pt locations, shape functions)

	  Ke.resize (dof_indices.size(), dof_indices.size());
	  Fe.resize (dof_indices.size());

	  for (unsigned int qp=0; qp<qrule.n_points(); qp++){ //loop over quad pts
	  
	  	//location of quadrature point
      const Real x = q_point[qp](0);
      const Real y = q_point[qp](1);
      
      //velocity field at quadrature point
      const Real xvel = 1.0;
      const Real yvel = 1.0;
      
	  	//integrate test functions (i) against trial functions (j)
      for (unsigned int i=0; i<phi.size(); i++){ 
        for (unsigned int j=0; j<phi.size(); j++){
          Ke(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]) //diffusion term
          	- JxW[qp]*(xvel*dphi[i][qp](0)*phi[j][qp] + yvel*dphi[i][qp](1)*phi[j][qp]); //convection term
          		//?!? this gives the right state but the sign doesn't seem right...??
        }
			}
			
      { //scope bubble
			//forcing function
      const Real fxy = -(x+y-2+x*x+y*y)*exp(-0.5*(x*x+y*y));
      //const Real fxy = 2;
      for (unsigned int i=0; i<phi.size(); i++)
        Fe(i) += JxW[qp]*fxy*phi[i][qp];
      } //end scope bubble
	  }

	  { //scope bubble
	  //enforce Dirichlet boundary conditions
		for (unsigned int side=0; side<elem->n_sides(); side++){
			if (elem->neighbor(side) == NULL){ //if this side if on boundary of domain
			  const std::vector<std::vector<Real> >&  phi_face = fe_face->get_phi();

			  const std::vector<Real>& JxW_face = fe_face->get_JxW();

			  const std::vector<Point >& qface_point = fe_face->get_xyz();

			  fe_face->reinit(elem, side);

			  for (unsigned int qp=0; qp<qface.n_points(); qp++){

			      const Real xf = qface_point[qp](0);
			      const Real yf = qface_point[qp](1);

			      const Real penalty = 1.e10;

			      const Real value = exact_solution(xf, yf);

			      for (unsigned int i=0; i<phi_face.size(); i++)
			        for (unsigned int j=0; j<phi_face.size(); j++)
			          Ke(i,j) += JxW_face[qp]*penalty*phi_face[i][qp]*phi_face[j][qp];

			      for (unsigned int i=0; i<phi_face.size(); i++)
			        Fe(i) += JxW_face[qp]*penalty*value*phi_face[i][qp];
		    }
			}
		}
	  } //end scope bubble


	  dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

	  system.matrix->add_matrix (Ke, dof_indices);
	  system.rhs->add_vector    (Fe, dof_indices);
	}

}
