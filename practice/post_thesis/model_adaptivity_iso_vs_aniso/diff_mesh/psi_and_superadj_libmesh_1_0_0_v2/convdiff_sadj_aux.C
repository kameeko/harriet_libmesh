#include "libmesh/getpot.h"

#include "convdiff_sadj_aux.h"

#include "libmesh/boundary_info.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_context.h"
#include "libmesh/mesh.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/zero_function.h"
#include "libmesh/equation_systems.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// System initialization
void ConvDiff_AuxSadjSys::init_data (){
	const unsigned int dim = this->get_mesh().mesh_dimension();

	//polynomial order and finite element type
	unsigned int conc_p = 1;
	GetPot infile("contamTrans.in");
	std::string fe_family = infile("fe_family", std::string("LAGRANGE"));

	// LBB needs better-than-quadratic velocities for better-than-linear
	// pressures, and libMesh needs non-Lagrange elements for
	// better-than-quadratic velocities.
	//libmesh_assert((conc_p == 1) || (fe_family != "LAGRANGE"));

	FEFamily fefamily = Utility::string_to_enum<FEFamily>(fe_family);
	aux_c_var = this->add_variable("sadj_aux_c", static_cast<Order>(conc_p), fefamily); 
	aux_zc_var = this->add_variable("sadj_aux_zc", static_cast<Order>(conc_p), fefamily);
	aux_fc_var = this->add_variable("sadj_aux_fc", static_cast<Order>(conc_p), fefamily);   

	//set parameters
  vx = infile("vx", 2.415e-5); // m/s
  react_rate = infile("reaction_rate", 0.0); // 1/s
  porosity = infile("porosity", 0.1); // (unitless)
  bsource = infile("bsource", -5.0); // ppb

  //compute dispersion tensor (assuming for now that velocity purely in x direction)
  double dlong = infile("dispersivity_longitudinal",60.0);
  double dtransh = infile("dispersivity_transverse_horizontal",6.0);
  double dtransv = infile("dispersivity_transverse_vertical",0.6);
  dispTens = NumberTensorValue(vx*dlong, 0.0, 0.0,
                              0.0, vx*dtransh, 0.0,
                              0.0, 0.0, vx*dtransv);  

  //regularization
  beta = infile("beta", 0.1);
  
	//indicate variables that change in time
	this->time_evolving(aux_c_var);
	this->time_evolving(aux_zc_var);
	this->time_evolving(aux_fc_var);
	
	// Useful debugging options
	// Set verify_analytic_jacobians to 1e-6 to use
	this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 0.);
	this->print_jacobians = infile("print_jacobians", false);
	this->print_element_jacobians = infile("print_element_jacobians", false);
	this->print_residuals = infile("print_residuals", false);
	this->print_solutions = infile("print_solutions", false);

	//set Dirichlet boundary conditions
  std::set<boundary_id_type> all_bdys;
  all_bdys.insert(0); all_bdys.insert(1); all_bdys.insert(2); all_bdys.insert(3); 
  if(dim == 3){
    all_bdys.insert(4); all_bdys.insert(5);
  }
  std::vector<unsigned int> just_f;
  just_f.push_back(aux_fc_var);
  ZeroFunction<Number> zero;
  this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, just_f, &zero)); //f=0 on boundary
  
  //influx side as Diri instead of flux BC
  std::vector<unsigned int> just_c; just_c.push_back(aux_c_var);
  std::vector<unsigned int> just_z; just_z.push_back(aux_zc_var);
  std::set<boundary_id_type> westside;
  if(dim == 2)
    westside.insert(3); 
  else if(dim == 3)
    westside.insert(4); 
  this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(westside, just_c, &zero));
  this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(westside, just_z, &zero));

	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();

}

// Context initialization
void ConvDiff_AuxSadjSys::init_context(DiffContext &context){
	FEMContext &ctxt = cast_ref<FEMContext&>(context);
  
	//stuff for things of pressure's family
	FEBase* c_elem_fe = NULL;

  ctxt.get_element_fe(aux_c_var, c_elem_fe);
  c_elem_fe->get_JxW();
  c_elem_fe->get_phi();
  c_elem_fe->get_dphi();
  c_elem_fe->get_xyz();

  FEBase* c_side_fe = NULL;
  ctxt.get_side_fe(aux_c_var, c_side_fe);

  c_side_fe->get_JxW();
  c_side_fe->get_phi();
  c_side_fe->get_dphi();
  c_side_fe->get_xyz();
  
  //add other solutions to the vectors that diff context should localize
  const System & sys = ctxt.get_system();
  NumericVector<Number> &primary_solution = 
 	 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_PrimarySys").solution;
 	NumericVector<Number> &aux_solution = 
 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_AuxSys").solution;
 	NumericVector<Number> &primary_sadj_solution = 
 	 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_PrimarySadjSys").solution;
 	ctxt.add_localized_vector(primary_solution, sys);
 	ctxt.add_localized_vector(aux_solution, sys);
 	ctxt.add_localized_vector(primary_sadj_solution, sys);
}

// Element residual and jacobian calculations
// Time dependent parts
bool ConvDiff_AuxSadjSys::element_time_derivative (bool request_jacobian, DiffContext& context){
	const unsigned int dim = this->get_mesh().mesh_dimension();
	Real PI = 3.14159265359;

	FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* c_elem_fe = NULL; 
  ctxt.get_element_fe( aux_c_var, c_elem_fe );

	// Element Jacobian * quadrature weights for interior integration
	const std::vector<Real> &JxW = c_elem_fe->get_JxW();

	const std::vector<std::vector<Real> >& phi = c_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dphi = c_elem_fe->get_dphi();
	
	// Physical location of the quadrature points
	const std::vector<Point>& qpoint = c_elem_fe->get_xyz();

	// The number of local degrees of freedom in each variable
	const unsigned int n_c_dofs = ctxt.get_dof_indices( aux_c_var ).size();

	// The subvectors and submatrices we need to fill:
	DenseSubMatrix<Number> &J_c_auxzc = ctxt.get_elem_jacobian(aux_c_var, aux_zc_var);
	DenseSubMatrix<Number> &J_c_auxc = ctxt.get_elem_jacobian(aux_c_var, aux_c_var);
	
	DenseSubMatrix<Number> &J_zc_auxc = ctxt.get_elem_jacobian(aux_zc_var, aux_c_var);
	DenseSubMatrix<Number> &J_zc_auxfc = ctxt.get_elem_jacobian(aux_zc_var, aux_fc_var);
	
	DenseSubMatrix<Number> &J_fc_auxfc = ctxt.get_elem_jacobian(aux_fc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_fc_auxzc = ctxt.get_elem_jacobian(aux_fc_var, aux_zc_var);
	
	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( aux_c_var );
	DenseSubVector<Number> &Rzc = ctxt.get_elem_residual( aux_zc_var );
	DenseSubVector<Number> &Rfc = ctxt.get_elem_residual( aux_fc_var );

	// Now we will build the element Jacobian and residual.
	// Constructing the residual requires the solution and its
	// gradient from the previous timestep.  This must be
	// calculated at each quadrature point by summing the
	// solution degree-of-freedom values by the appropriate
	// weight functions.
	unsigned int n_qpoints = ctxt.get_element_qrule().n_points();
	
  const System & sys = ctxt.get_system();
  NumericVector<Number> &primary_solution = 
 	 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_PrimarySys").solution;
 	NumericVector<Number> &aux_solution = 
 	 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_AuxSys").solution;
 	NumericVector<Number> &primary_sadj_solution = 
 	 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_PrimarySadjSys").solution;
  std::vector<Number> c_at_qp (n_qpoints, 0);
  std::vector<Number> zc_at_qp (n_qpoints, 0);
  std::vector<Number> aux_c_at_qp (n_qpoints, 0);
  std::vector<Number> aux_zc_at_qp (n_qpoints, 0);
  std::vector<Number> aux_fc_at_qp (n_qpoints, 0);
  std::vector<Gradient> grad_aux_c_at_qp (n_qpoints, 0);
  std::vector<Gradient> grad_aux_zc_at_qp (n_qpoints, 0);
  std::vector<Gradient> grad_aux_fc_at_qp (n_qpoints, 0);
  std::vector<Number> sadj_c_at_qp (n_qpoints, 0);
  std::vector<Number> sadj_zc_at_qp (n_qpoints, 0);
  unsigned int primal_c_var = sys.get_equation_systems().get_system("ConvDiff_PrimarySys").variable_number("c");
  unsigned int primal_zc_var = sys.get_equation_systems().get_system("ConvDiff_PrimarySys").variable_number("zc");
  unsigned int primal_auxc_var = sys.get_equation_systems().get_system("ConvDiff_AuxSys").variable_number("aux_c");
  unsigned int primal_auxzc_var = sys.get_equation_systems().get_system("ConvDiff_AuxSys").variable_number("aux_zc");
  unsigned int primal_auxfc_var = sys.get_equation_systems().get_system("ConvDiff_AuxSys").variable_number("aux_fc");
  unsigned int sadj_c_var = sys.get_equation_systems().get_system("ConvDiff_PrimarySadjSys").variable_number("sadj_c");
  unsigned int sadj_zc_var = sys.get_equation_systems().get_system("ConvDiff_PrimarySadjSys").variable_number("sadj_zc");

  ctxt.interior_values<Number>(primal_c_var, primary_solution, c_at_qp); 
  ctxt.interior_values<Number>(primal_zc_var, primary_solution, zc_at_qp);
  ctxt.interior_values<Number>(primal_auxc_var, aux_solution, aux_c_at_qp); 
  ctxt.interior_values<Number>(primal_auxzc_var, aux_solution, aux_zc_at_qp);
  ctxt.interior_values<Number>(primal_auxfc_var, aux_solution, aux_fc_at_qp);
  ctxt.interior_gradients<Gradient>(primal_auxc_var, aux_solution, grad_aux_c_at_qp); 
  ctxt.interior_gradients<Gradient>(primal_auxzc_var, aux_solution, grad_aux_zc_at_qp);
  ctxt.interior_gradients<Gradient>(primal_auxfc_var, aux_solution, grad_aux_fc_at_qp);
  ctxt.interior_values<Number>(sadj_c_var, primary_sadj_solution, sadj_c_at_qp); 
  ctxt.interior_values<Number>(sadj_zc_var, primary_sadj_solution, sadj_zc_at_qp);
  
	for (unsigned int qp=0; qp != n_qpoints; qp++)
	  {
	    Number 
	      auxc = ctxt.interior_value(aux_c_var, qp),
	      auxzc = ctxt.interior_value(aux_zc_var, qp),
	      auxfc = ctxt.interior_value(aux_fc_var, qp);
	    Gradient 
	      grad_auxc = ctxt.interior_gradient(aux_c_var, qp),
	      grad_auxzc = ctxt.interior_gradient(aux_zc_var, qp),
	      grad_auxfc = ctxt.interior_gradient(aux_fc_var, qp);
	      
	    Number primal_c = c_at_qp[qp];
	    Number primal_zc = zc_at_qp[qp];
	    Number primal_auxc = aux_c_at_qp[qp];
	    Number primal_auxzc = aux_zc_at_qp[qp];
	    Number primal_auxfc = aux_fc_at_qp[qp];
	    Gradient primal_grad_auxc = grad_aux_c_at_qp[qp];
	    Gradient primal_grad_auxzc = grad_aux_zc_at_qp[qp];
	    Gradient primal_grad_auxfc = grad_aux_fc_at_qp[qp];
	    Number sadj_c = sadj_c_at_qp[qp];
	    Number sadj_zc = sadj_zc_at_qp[qp];
	    
	  	//location of quadrature point
	  	const Real ptx = qpoint[qp](0);
	  	const Real pty = qpoint[qp](1);
	  	const Real ptz = qpoint[qp](2);
			
			Real R = react_rate; 
			NumberTensorValue k = porosity*dispTens;
			NumberVectorValue U(porosity*vx, 0.0, 0.0);

			// First, an i-loop over the  degrees of freedom.
			for (unsigned int i=0; i != n_c_dofs; i++){
     		
				Rc(i) += JxW[qp]*(-k*grad_auxzc*dphi[i][qp] + U*grad_auxzc*phi[i][qp] 
	      	- 2.*R*primal_zc*auxc*phi[i][qp] - 2.*R*auxzc*primal_c*phi[i][qp]
	      	- 2.*R*sadj_zc*primal_auxc*phi[i][qp] - 2.*R*primal_auxzc*sadj_c*phi[i][qp]); 
	      Rzc(i) += JxW[qp]*(-k*grad_auxc*dphi[i][qp] - U*grad_auxc*phi[i][qp] 
	      						+ auxfc*phi[i][qp] - 2.*R*primal_c*auxc*phi[i][qp] - 2.*R*sadj_c*primal_auxc*phi[i][qp]);
   			Rfc(i) += JxW[qp]*(auxzc*phi[i][qp] + beta*grad_auxfc*dphi[i][qp]);
        
        //rhs of adjoint
				Rc(i) -= JxW[qp]*(-k*primal_grad_auxzc*dphi[i][qp] + U*primal_grad_auxzc*phi[i][qp] 
	      	- 2.*R*primal_zc*primal_auxc*phi[i][qp] - 2.*R*primal_auxzc*primal_c*phi[i][qp]);
				if((qoi_option == 0) ||
          (qoi_option == 1 && 
      			(dim == 3 && (fabs(ptx - 1150.) <= 50. && fabs(pty - 825.) <= 50. && ptz >= 80.))) ||
      		(qoi_option == 1 && 
      			(dim == 2 && (fabs(ptx - 1150.) <= 50. && fabs(pty - 825.) <= 50.))) 	){			
	      		
					Rc(i) -= JxW[qp]*phi[i][qp]; 
				}
				Rzc(i) -= JxW[qp]*(-k*primal_grad_auxc*dphi[i][qp] - U*primal_grad_auxc*phi[i][qp] 
					  						+ primal_auxfc*phi[i][qp] - 2.*R*primal_c*primal_auxc*phi[i][qp]);	
				Rfc(i) -= JxW[qp]*(primal_auxzc*phi[i][qp] + beta*primal_grad_auxfc*dphi[i][qp]);
        

				if (request_jacobian){
					for (unsigned int j=0; j != n_c_dofs; j++){
        		J_c_auxzc(i,j) += JxW[qp]*(-k*dphi[j][qp]*dphi[i][qp] + U*dphi[j][qp]*phi[i][qp] - 2.*R*phi[j][qp]*primal_c*phi[i][qp]);
						J_c_auxc(i,j) += JxW[qp]*(-2.*R*primal_zc*phi[j][qp]*phi[i][qp]);
						
						J_zc_auxc(i,j) += JxW[qp]*(-k*dphi[j][qp]*dphi[i][qp] - U*dphi[j][qp]*phi[i][qp]
																- 2.*R*primal_c*phi[j][qp]*phi[i][qp]);
						J_zc_auxfc(i,j) += JxW[qp]*(phi[j][qp]*phi[i][qp]);
					
						J_fc_auxzc(i,j) += JxW[qp]*(phi[j][qp])*phi[i][qp];
				  	J_fc_auxfc(i,j) += JxW[qp]*(beta*dphi[j][qp]*dphi[i][qp]);
		      	
					} // end of the inner dof (j) loop
			  } // end - if (compute_jacobian && context.get_elem_solution_derivative())

			} // end of the outer dof (i) loop

    } // end of the quadrature point (qp) loop
    
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	Point data_point = datapts[dnum];
	  	if(dataelems[dnum] == ctxt.get_elem().id()){
	  		
	  		Number auxc_pointy = ctxt.point_value(aux_c_var, data_point);
	  		Number primal_auxc_pointy = primal_auxc_vals[dnum];
	  		
	  		unsigned int dim = ctxt.get_system().get_mesh().mesh_dimension();
		    FEType fe_type = ctxt.get_element_fe(aux_c_var)->get_fe_type();
		    
		    //go between physical and reference element
		    Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
		    
        std::vector<Real> point_phi(n_c_dofs);
      	for (unsigned int i=0; i != n_c_dofs; i++){
      		//get value of basis function at mapped point in reference (master) element
          point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
        }
        
        for (unsigned int i=0; i != n_c_dofs; i++){
  	  		Rc(i) += auxc_pointy*point_phi[i];
  	  		
  	  		//rhs of adjoint
  	  		Rc(i) -= primal_auxc_pointy*point_phi[i];
	  
					if (request_jacobian){
						for (unsigned int j=0; j != n_c_dofs; j++){
							J_c_auxc(i,j) += point_phi[j]*point_phi[i];
						}
				  }
	  
  			}
	  	}
	  }

	return request_jacobian;
}

//for non-Dirichlet boundary conditions and the bit from diffusion term
bool ConvDiff_AuxSadjSys::side_time_derivative(bool request_jacobian, DiffContext & context)
{
  const unsigned int dim = this->get_mesh().mesh_dimension();

  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEBase* side_fe = NULL;
  ctxt.get_side_fe(aux_c_var, side_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = side_fe->get_JxW();

  // Side basis functions
  const std::vector<std::vector<Real> > &phi = side_fe->get_phi();

  // Side Quadrature points
  const std::vector<Point > &qside_point = side_fe->get_xyz();

  //normal vector
  const std::vector<Point> &face_normals = side_fe->get_normals();

  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices(aux_c_var).size();

  // The subvectors and submatrices we need to fill:
  DenseSubMatrix<Number> &J_c_auxz = ctxt.get_elem_jacobian(aux_c_var, aux_zc_var);
  
  DenseSubVector<Number> &Rc = ctxt.get_elem_residual( aux_c_var );
  //Rf gets no contribution from sides

  unsigned int n_qpoints = ctxt.get_side_qrule().n_points();

  bool isEast = false;
  if (dim == 3){
    isEast = ctxt.has_side_boundary_id(2);
  }
  else if (dim == 2){
    isEast = ctxt.has_side_boundary_id(1);
  }
  
  //set (in)flux boundary condition on west side
  //homogeneous neumann (Danckwerts) outflow boundary condition on east side
  //no-flux (equivalently, homoegenous neumann) boundary conditions on north, south, top, bottom sides
  //"strong" enforcement of boundary conditions
  for (unsigned int qp=0; qp != n_qpoints; qp++)
  {
    Number aux_z = ctxt.side_value(aux_zc_var, qp);

    //velocity vector
    NumberVectorValue U(porosity*vx, 0., 0.); 

    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      if(isEast)
        Rc(i) += JxW[qp]*(-U*face_normals[qp]*aux_z)*phi[i][qp];
      
      if(request_jacobian && context.get_elem_solution_derivative())
      {
        for (unsigned int j=0; j != n_c_dofs; j++)
        {
          if(isEast)
            J_c_auxz(i,j) += JxW[qp]*(-U*face_normals[qp]*phi[j][qp])*phi[i][qp];
        }
      } // end - if (request_jacobian && context.get_elem_solution_derivative())
    } //end of outer dof (i) loop
  }

  return request_jacobian;
}

