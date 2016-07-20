#include "libmesh/getpot.h"

#include "contamTrans_sys.h"

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
#include "libmesh/elem.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

//system initialization
void ContamTransSys::init_data(){
  const unsigned int dim = this->get_mesh().mesh_dimension();

  GetPot infile("general.in");
  unsigned int poly_order = infile("poly_order",1);
  std::string fefamily = infile("fe_family", std::string("LAGRANGE"));

  p_var = this->add_variable("p", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));
  z_var = this->add_variable("z", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));
  k_var = this->add_variable("k", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));

  //indicate variables that change in time
  this->time_evolving(p_var);
  this->time_evolving(z_var);
  this->time_evolving(k_var);

  // Useful debugging options'
	// Set verify_analytic_jacobians to positive to use
	this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 0.);
	this->print_jacobians = infile("print_jacobians", false);
	this->print_element_jacobians = infile("print_element_jacobians", false);

	//set Dirichlet boundary conditions
	std::set<boundary_id_type> east_bdy;
	std::set<boundary_id_type> west_bdy;
	std::set<boundary_id_type> eastwest_bdy;
	if (dim == 3){
    west_bdy.insert(4);
    east_bdy.insert(2);
    eastwest_bdy.insert(2); eastwest_bdy.insert(4);
  }
  else if (dim == 2){
    west_bdy.insert(3);
    east_bdy.insert(1);
    eastwest_bdy.insert(1); eastwest_bdy.insert(3);
  }
  std::vector<unsigned int> p_set;
  p_set.push_back(p_var);
  std::vector<unsigned int> z_set;
  z_set.push_back(z_var);
  //Real westPressure = 2.606e5; //Pa
  Real westPressure = 2.606e2; //kPa
  if(infile("do_square",true))
    westPressure *= 1./46.;
  ConstFunction<Number> WestPressure(westPressure);
  ConstFunction<Number> EastPressure(0.0);
  ZeroFunction<Number> homoDiri;
  this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(west_bdy, p_set, &WestPressure));
  this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(east_bdy, p_set, &EastPressure));
  this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(eastwest_bdy, z_set, &homoDiri));

	//set parameters
  //avg_perm = 2.72e-10; //m^2
  avg_perm = 275.8; //darcies
  beta1 = infile("beta1",1.e-6);
  beta2 = infile("beta2",1.e-6);
  
  //DEBUG
  //std::vector<unsigned int> k_set;
  //k_set.push_back(k_var);
  //ConstFunction<Number> eep(avg_perm);
  //std::set<boundary_id_type> all_bdy;
  //all_bdy.insert(0); all_bdy.insert(1); all_bdy.insert(2); all_bdy.insert(3); 
  //this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdy, k_set, &eep));

	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

//context initialization
void ContamTransSys::init_context(DiffContext & context){
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* p_elem_fe;

  ctxt.get_element_fe(p_var, p_elem_fe);

	p_elem_fe->get_JxW();
	p_elem_fe->get_phi();
	p_elem_fe->get_dphi();
	p_elem_fe->get_xyz();
}

//element residual and jacobian calculations
bool ContamTransSys::element_time_derivative(bool request_jacobian, DiffContext & context)
{
  const unsigned int dim = this->get_mesh().mesh_dimension();

  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* p_elem_fe = NULL;
  ctxt.get_element_fe( p_var, p_elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = p_elem_fe->get_JxW();

  const std::vector<std::vector<Real> >& phi = p_elem_fe->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = p_elem_fe->get_dphi();

  // Physical location of the quadrature points
  const std::vector<Point>& qpoint = p_elem_fe->get_xyz();

  // The number of local degrees of freedom in each variable
  const unsigned int n_p_dofs = ctxt.get_dof_indices( p_var ).size();

  // The subvectors and submatrices we need to fill:
  DenseSubMatrix<Number> &J_p_p = ctxt.get_elem_jacobian(p_var, p_var);
  DenseSubMatrix<Number> &J_p_z = ctxt.get_elem_jacobian(p_var, z_var);
  DenseSubMatrix<Number> &J_p_k = ctxt.get_elem_jacobian(p_var, k_var);
  
  DenseSubMatrix<Number> &J_z_p = ctxt.get_elem_jacobian(z_var, p_var);
  DenseSubMatrix<Number> &J_z_k = ctxt.get_elem_jacobian(z_var, k_var);
  
  DenseSubMatrix<Number> &J_k_p = ctxt.get_elem_jacobian(k_var, p_var);
  DenseSubMatrix<Number> &J_k_z = ctxt.get_elem_jacobian(k_var, z_var);
  DenseSubMatrix<Number> &J_k_k = ctxt.get_elem_jacobian(k_var, k_var);
  
  DenseSubVector<Number> &Rp = ctxt.get_elem_residual(p_var);
  DenseSubVector<Number> &Rz = ctxt.get_elem_residual(z_var);
  DenseSubVector<Number> &Rk = ctxt.get_elem_residual(k_var);

  // Now we will build the element Jacobian and residual.
  // Constructing the residual requires the solution and its
  // gradient from the previous timestep.  This must be
  // calculated at each quadrature point by summing the
  // solution degree-of-freedom values by the appropriate
  // weight functions.
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  for (unsigned int qp=0; qp != n_qpoints; qp++)
  {
    Number 
      k = ctxt.interior_value(k_var, qp);
    Gradient 
      grad_p = ctxt.interior_gradient(p_var, qp),
      grad_z = ctxt.interior_gradient(z_var, qp),
      grad_k = ctxt.interior_gradient(k_var, qp);
    
    // First, an i-loop over the  degrees of freedom.
    for (unsigned int i=0; i != n_p_dofs; i++)
    {
      Rp(i) += JxW[qp]*(-k*grad_z*dphi[i][qp]);
      Rz(i) += JxW[qp]*(k*grad_p*dphi[i][qp]);
      Rk(i) += JxW[qp]*(beta1*(k-avg_perm)*phi[i][qp] + beta2*grad_k*dphi[i][qp] - grad_p*grad_z*phi[i][qp]);
      
      if (request_jacobian && ctxt.get_elem_solution_derivative())
      {
	      for (unsigned int j=0; j != n_p_dofs; j++)
	      {
	        J_p_z(i,j) += JxW[qp]*(-k*dphi[j][qp]*dphi[i][qp]);
	        J_p_k(i,j) += JxW[qp]*(-phi[j][qp]*grad_z*dphi[i][qp]);
	      
	        J_z_p(i,j) += JxW[qp]*(k*dphi[j][qp]*dphi[i][qp]); 
	        J_z_k(i,j) += JxW[qp]*(phi[j][qp]*grad_p*dphi[i][qp]);
	        
	        J_k_p(i,j) += JxW[qp]*(-dphi[j][qp]*grad_z)*phi[i][qp];
	        J_k_z(i,j) += JxW[qp]*(-grad_p*dphi[j][qp])*phi[i][qp];
	        J_k_k(i,j) += JxW[qp]*(beta1*phi[j][qp]*phi[i][qp] + beta2*dphi[j][qp]*dphi[i][qp]);
	      } // end of the inner dof (j) loop
      } // end - if (request_jacobian && context.get_elem_solution_derivative())
    } // end of the outer dof (i) loop
  } // end of the quadrature point (qp) loop
  
  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
    Point data_point = datapts[dnum];
    if(dataelems[dnum] == ctxt.get_elem().id()){

      Number ppred = ctxt.point_value(p_var, data_point);
      Number pstar = datavals[dnum];

      unsigned int dim = ctxt.get_system().get_mesh().mesh_dimension();
      FEType fe_type = ctxt.get_element_fe(p_var)->get_fe_type();
      
      //go between physical and reference element
      Point p_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point);   
      
      std::vector<Real> point_phi(n_p_dofs);
      for (unsigned int i=0; i != n_p_dofs; i++){
        //get value of basis function at mapped point in reference (master) element
        point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, p_master); 
      }
      
      for (unsigned int i=0; i != n_p_dofs; i++){
        Rp(i) += (ppred - pstar)*point_phi[i];
  
        if (request_jacobian){
          for (unsigned int j=0; j != n_p_dofs; j++)
            J_p_p(i,j) += point_phi[j]*point_phi[i] ;
        }
  
      }
    }
  }

  return request_jacobian;
}

//generate data
void ContamTransSys::postprocess(){

  //reset computed QoIs
  computed_QoI[0] = 0.0;

  FEMSystem::postprocess();

  this->comm().sum(computed_QoI[0]);

}

