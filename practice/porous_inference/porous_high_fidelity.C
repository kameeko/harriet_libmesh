/* This is where we define the assembly of the contaminant system */

// General libMesh includes
#include "libmesh/getpot.h"
#include "libmesh/fe_base.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/parallel.h"
#include "libmesh/fem_context.h"
#include "libmesh/fe_interface.h"
#include "libmesh/periodic_boundaries.h"
#include "libmesh/periodic_boundary.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/zero_function.h"
#include "libmesh/dof_map.h"

// Local includes
#include "porous_high_fidelity.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

void PorousHFSystem::init_data ()
{  
  GetPot infile("porous_high_fidelity.in");
 
  // Input parameters  
  // Regularization parameter
  alpha = infile("alpha", 1.0);
      
  // Approximation basis and order for both c_m and c_i_m
  std::string fe_family = infile("fe_family", std::string("LAGRANGE"));  

  // Useful debugging options
  // Set verify_analytic_jacobians to 1e-6 to use
  this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 1.e-6);
  this->print_jacobians = infile("print_jacobians", false);
  this->print_element_jacobians = infile("print_element_jacobians", false);
  
  // Create the variables c_m and c_im to be approximated by the user specified basis  
  FEFamily fefamily = Utility::string_to_enum<FEFamily>(fe_family);
  
  // Add the system variables "K_var", "p_var" and "z_var"
  K_var = this->add_variable ("K", static_cast<Order>(FIRST), //parameter
			      fefamily);
  p_var = this->add_variable ("p",static_cast<Order>(FIRST), //state
			      fefamily);
  z_var = this->add_variable ("z",static_cast<Order>(FIRST), //adjoint
			      fefamily);

  // Tell the system to march both K_var, p_var and z_var forward in time
  this->time_evolving(K_var);  
  this->time_evolving(p_var); 
  this->time_evolving(z_var); 

  // Now we apply the boundary conditions
   
  // Apply the Dirichlet boundary condition on the left hand side y boundary
  // Supply the relevant boundary id to a set
  const boundary_id_type left_y_id = 3;
  std::set<boundary_id_type> left_y_bdy;
  left_y_bdy.insert(left_y_id);

  const boundary_id_type right_y_id = 1;
  std::set<boundary_id_type> right_y_bdy;
  right_y_bdy.insert(right_y_id);

  const boundary_id_type bottom_x_id = 0;
  const boundary_id_type top_x_id = 2;
  std::set<boundary_id_type> bottom_top_bdys;
  bottom_top_bdys.insert(bottom_x_id);
  bottom_top_bdys.insert(top_x_id);
   
  // Variables on which the boundary condition is to be applied
  std::vector<unsigned int> p_only(1, p_var);
  std::vector<unsigned int> z_only(2, z_var);
  std::vector<unsigned int> p_z(1, p_var);
  p_z.push_back(z_var);

  // Boundary conditions to be applied
  ZeroFunction<Number> zero;
  ConstFunction<Number> one(1);
  
  // Apply the Dirichlet boundary condition to specifed variables on the given set of boundary ids
  // On the left bdry apply the boundary conditions p = 1, z = 0
  this->get_dof_map().add_dirichlet_boundary
  (DirichletBoundary (left_y_bdy, p_only, &one));
  this->get_dof_map().add_dirichlet_boundary
  (DirichletBoundary (left_y_bdy, z_only, &zero));

  // On the right bdry apply the boundary conditions p = 0, z = 0
  this->get_dof_map().add_dirichlet_boundary
  (DirichletBoundary (right_y_bdy, p_z, &zero));

  // On the top and bottom bdrys apply the condition z = 0
  //this->get_dof_map().add_dirichlet_boundary
  //(DirichletBoundary (bottom_top_bdys, z_only, &zero));

  // Remaining bcs, pdvn{K} = 0 on all bdrys, pdvn{p} = 0 on top and bottom bdrys
                
  // Do the parent's initialization after variables are defined
  FEMSystem::init_data();

}

void PorousHFSystem::init_context(DiffContext &context)
{
  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  // Get element and side basis functions for the permeability variable K
  FEBase* K_elem_fe = NULL;

  c.get_element_fe(K_var, K_elem_fe);
  K_elem_fe->get_JxW();
  K_elem_fe->get_phi();
  K_elem_fe->get_dphi();

  FEBase* K_side_fe = NULL;
  c.get_side_fe(K_var, K_side_fe);

  K_side_fe->get_JxW();
  K_side_fe->get_phi();
  K_side_fe->get_dphi();

  // Get element and side basis functions for the pressure variable p
  FEBase* p_elem_fe = NULL;

  c.get_element_fe(p_var, p_elem_fe);
  p_elem_fe->get_JxW();
  p_elem_fe->get_phi();
  p_elem_fe->get_dphi();

  FEBase* p_side_fe = NULL;
  c.get_side_fe(p_var, p_side_fe);

  p_side_fe->get_JxW();
  p_side_fe->get_phi();
  p_side_fe->get_dphi();

}

bool PorousHFSystem::element_time_derivative (bool request_jacobian,
                                            DiffContext &context)
{
  // Are the jacobians specified analytically ?
  bool compute_jacobian = request_jacobian && _analytic_jacobians;

  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEBase* K_elem_fe = NULL; //initializing pointer
  c.get_element_fe( K_var, K_elem_fe );
  FEBase* p_elem_fe = NULL; //initializing pointer; same for z
  c.get_element_fe( p_var, p_elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = K_elem_fe->get_JxW();

  // Element basis functions - what about z? or is that supposed to have the same as p? ***********
  const std::vector<std::vector<Real> > &phi_K = K_elem_fe->get_phi();
  const std::vector<std::vector<Real> > &phi_p = p_elem_fe->get_phi();
  const std::vector<std::vector<RealGradient> > &dphi_p = p_elem_fe->get_dphi();

  // The number of local degrees of freedom in each variable
  const unsigned int n_K_dofs = c.get_dof_indices(K_var).size();
  const unsigned int n_p_dofs = c.get_dof_indices(p_var).size();

  // The subvectors and submatrices we need to fill:
  DenseSubMatrix<Number> &KKK = c.get_elem_jacobian(K_var,K_var);
  DenseSubMatrix<Number> &KKp = c.get_elem_jacobian(K_var,p_var);
  DenseSubMatrix<Number> &KKz = c.get_elem_jacobian(K_var,z_var);

  DenseSubMatrix<Number> &KpK = c.get_elem_jacobian(p_var,K_var);
  DenseSubMatrix<Number> &Kpp = c.get_elem_jacobian(p_var,p_var);
  DenseSubMatrix<Number> &Kpz = c.get_elem_jacobian(p_var,z_var);

  DenseSubMatrix<Number> &KzK = c.get_elem_jacobian(z_var,K_var);
  DenseSubMatrix<Number> &Kzp = c.get_elem_jacobian(z_var,p_var);
  DenseSubMatrix<Number> &Kzz = c.get_elem_jacobian(z_var,z_var);
  
  DenseSubVector<Number> &FK = c.get_elem_residual(K_var);
  DenseSubVector<Number> &Fp = c.get_elem_residual(p_var);
  DenseSubVector<Number> &Fz = c.get_elem_residual(z_var);

  // Now we will build the element Jacobian and residual.
  // Constructing the residual requires the solution and its
  // gradient from the previous timestep.  This must be
  // calculated at each quadrature point by summing the
  // solution degree-of-freedom values by the appropriate
  // weight functions.
  unsigned int n_qpoints = c.get_element_qrule().n_points();

  // Physical location of the quadrature points
  const std::vector<Point>& qpoint = K_elem_fe->get_xyz();

  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {
      // Location of this qp
      Number x = qpoint[qp](0);
      Number y = qpoint[qp](1);

      // Compute the solution and solution gradient at the Newton iterate
      Number K_value = c.interior_value(K_var, qp);
      // We need to ensure that K never has a negative value 
      //K_value = fabs(K_value);
      Number p_value = c.interior_value(p_var, qp);
      Number z_value = c.interior_value(z_var, qp);
                  
      Gradient grad_p = c.interior_gradient(p_var, qp);
      Gradient grad_z = c.interior_gradient(z_var, qp);
                              
      // The residual contribution from this element

      // The parameter equation contribution
      for (unsigned int i=0; i != n_K_dofs; i++)
	{	  
	  FK(i) += JxW[qp] * (alpha*(pow(x,2.)+pow(y,2.)) - alpha*(K_value) - (grad_p*grad_z) )*phi_K[i][qp] ; 
	}
      
      // State and adjoint equation contributions
      for (unsigned int i=0; i != n_p_dofs; i++)
	{	  
	  Fp(i) += JxW[qp] * (-K_value*(grad_z*dphi_p[i][qp]) ) ; 

	  Fz(i) += JxW[qp] * (-K_value*(grad_p*dphi_p[i][qp]) ) ; 
	}
      
      if (compute_jacobian)
        for (unsigned int i=0; i != n_K_dofs; i++)
	  {
	    for (unsigned int j=0; j != n_K_dofs; ++j)
	      {
		KKK(i,j) += JxW[qp] * (-alpha * phi_K[j][qp] * phi_K[i][qp] ); // Parameter-Parameter contribution
	      }
	    for (unsigned int j=0; j != n_p_dofs; ++j)
	      {
		KKp(i,j) += JxW[qp] * (-dphi_p[j][qp]*grad_z)*phi_K[i][qp]; // Parameter-State contribution
		
		KKz(i,j) += JxW[qp] * (-grad_p*dphi_p[j][qp])*phi_K[i][qp]; // Paramater-Adjoint contribution
	      }
	  }

      if (compute_jacobian)
        for (unsigned int i=0; i != n_p_dofs; i++)
	  {
	    for (unsigned int j=0; j != n_K_dofs; ++j)
	      {
		// The analytic jacobian
		KpK(i,j) += JxW[qp] * (-phi_K[j][qp] * (grad_z*dphi_p[i][qp])); // State-Parameter contribution

		KzK(i,j) += JxW[qp] * (-phi_K[j][qp] * (grad_p*dphi_p[i][qp])); // Adjoint-Parameter contribution
	      }
	  }

      if (compute_jacobian)
        for (unsigned int i=0; i != n_p_dofs; i++)
	  {
	    for (unsigned int j=0; j != n_p_dofs; ++j)
	      {
		// The analytic jacobian				
		
		Kpz(i,j) += JxW[qp] * (-K_value * (dphi_p[j][qp]*dphi_p[i][qp])); // State-Adjoint contribution
		
		Kzp(i,j) += JxW[qp] * (-K_value * (dphi_p[j][qp]*dphi_p[i][qp])); // Adjoint-State contribution
	      }
	  }
      
    } // end of the quadrature point qp-loop

  // // Additional point contributions to the stiffness matrix
  const Point data_point_1(0.25, 0.5);
  const Point data_point_2(0.5, 0.25);
  const Point data_point_3(0.75, 0.5);
  const Point data_point_4(0.5, 0.75);

  if( c.get_elem().contains_point(data_point_1))
    {
      const Real data_1 = 0.75;
      
      Number p_value = c.point_value(p_var, data_point_1);      

      unsigned int dim = get_mesh().mesh_dimension();
      FEType fe_type = p_elem_fe->get_fe_type();
      Point p_master = FEInterface::inverse_map(dim, fe_type, &c.get_elem(), data_point_1); 
      	//going between physical and reference element  

      std::vector<Real> point_phi(n_p_dofs);
      for (unsigned int i=0; i != n_p_dofs; i++)
        {
          point_phi[i] = FEInterface::shape(dim, fe_type, &c.get_elem(), i, p_master); //get value of basis function at mapped point in reference (master) element
        }
      
      for (unsigned int i=0; i != n_p_dofs; i++)        
  	{
  	  Fp(i) += (data_1 - p_value) * point_phi[i];
	  
  	  if (compute_jacobian)                        
  	      {
  		for (unsigned int j=0; j != n_p_dofs; j++)
  		  {
  		    Kpp(i,j) += -point_phi[j]*point_phi[i] ; // Parameter-Parameter contribution	
  		  }
  	      }
	  
  	}
    }

  if( c.get_elem().contains_point(data_point_2))
    {
      const Real data_2 = 0.5;
      
      Number p_value = c.point_value(p_var, data_point_2);      

      unsigned int dim = get_mesh().mesh_dimension();
      FEType fe_type = p_elem_fe->get_fe_type();
      Point p_master = FEInterface::inverse_map(dim, fe_type, &c.get_elem(), data_point_2);

      std::vector<Real> point_phi(n_p_dofs);
      for (unsigned int i=0; i != n_p_dofs; i++)
        {
          point_phi[i] = FEInterface::shape(dim, fe_type, &c.get_elem(), i, p_master);
        }

      for (unsigned int i=0; i != n_p_dofs; i++)        
  	{
  	  Fp(i) += (data_2 - p_value) * point_phi[i];
	  
  	  if (compute_jacobian)                        
  	      {
  		for (unsigned int j=0; j != n_p_dofs; j++)
  		  {
  		    Kpp(i,j) += -point_phi[j]*point_phi[i] ; // Parameter-Parameter contribution       
  		  }
  	      }	  
  	}
    }

  if( c.get_elem().contains_point(data_point_3))
    {
      const Real data_3 = 0.25;
      
      Number p_value = c.point_value(p_var, data_point_3);      

      unsigned int dim = get_mesh().mesh_dimension();
      FEType fe_type = p_elem_fe->get_fe_type();
      Point p_master = FEInterface::inverse_map(dim, fe_type, &c.get_elem(), data_point_3);

      std::vector<Real> point_phi(n_p_dofs);
      for (unsigned int i=0; i != n_p_dofs; i++)
        {
          point_phi[i] = FEInterface::shape(dim, fe_type, &c.get_elem(), i, p_master);
        }

      for (unsigned int i=0; i != n_p_dofs; i++)        
  	{
  	  Fp(i) += (data_3 - p_value) * point_phi[i];
	  
  	  if (compute_jacobian)                        
  	      {
  		for (unsigned int j=0; j != n_p_dofs; j++)
  		  {
  		    Kpp(i,j) += -point_phi[j]*point_phi[i] ; // Parameter-Parameter contribution       
  		  }
  	      }	  
  	}                  
      
    }

  if( c.get_elem().contains_point(data_point_4))
    {
      const Real data_4 = 0.5;
      
      Number p_value = c.point_value(p_var, data_point_4);      

      unsigned int dim = get_mesh().mesh_dimension();
      FEType fe_type = p_elem_fe->get_fe_type();
      Point p_master = FEInterface::inverse_map(dim, fe_type, &c.get_elem(), data_point_4);

      std::vector<Real> point_phi(n_p_dofs);
      for (unsigned int i=0; i != n_p_dofs; i++)
        {
          point_phi[i] = FEInterface::shape(dim, fe_type, &c.get_elem(), i, p_master);
        }
      
      for (unsigned int i=0; i != n_p_dofs; i++)        
  	{
  	  Fp(i) += (data_4 - p_value) * point_phi[i];
	  
  	  if (compute_jacobian)                        
  	      {
  		for (unsigned int j=0; j != n_p_dofs; j++)
  		  {
  		    Kpp(i,j) += -point_phi[j]*point_phi[i] ; // Parameter-Parameter contribution       
  		  }
  	      }	  
  	}              
    }

  return compute_jacobian;
}

void PorousHFSystem::postprocess()
{
  computed_QoI[0] = 0.0;

  FEMSystem::postprocess();

  this->comm().sum(computed_QoI[0]);
}

// // The Permeability function
// // K(x,y) = 1 in immobile region
// //        = 10 in mobile region
// Number PorousHFSystem::permeability(Real x, Real y)
// {
//   Point p;

//   Real a,b;
  
//   // FIX THIS: Hack for point size
//   for(int i = 0; i <= 12; i++)
//     {
//       p = centres[i];
//       a = p(0);
//       b = p(1);

//       if((y <= x + b - a + L/2.) || (y <= -x + b + a + L/2.) || (y >= -x + a + b - L/2.) || (y >= x + b - a - L/2.))
// 	{
// 	  return 1;
// 	}
//     }

//   return 10.;
// }
  
// Real a = 1./4., b = 0.;
  
// if((0 < x) && (x <= a))
//   {
//     if((y >= -x + a + b) && (y <= x + (b - a) + 1./2.))
//       {
// 	return 1;
//       }
//   }
//  else if((a < x) && (x <= a + 1./4.))
//    {
//      if((y >= x + b - a) && (y <= -x + a + b + 1./2.))
//        {
// 	 return 1;
//        }
//    }

// a = 3./4., b = 0.;

// if((0 < x) && (x <= a))
//   {
//     if((y >= -x + a + b) && (y <= x + (b - a) + 1./2.))
//       {
// 	return 1;
//       }
//   }
//  else if((a < x) && (x <= a + 1./4.))
//    {
//      if((y >= x + b - a) && (y <= -x + a + b + 1./2.))
//        {
// 	 return 1;
//        }
//    }

// a = 3./4., b = 1./2.;
  
// if((0 < x) && (x <= a))
//   {
//     if((y >= -x + a + b) && (y <= x + (b - a) + 1./2.))
//       {
// 	return 1;
//       }
//   }
//  else if((a < x) && (x <= a + 1./4.))
//    {
//      if((y >= x + b - a) && (y <= -x + a + b + 1./2.))
//        {
// 	 return 1;
//        }
//    }

// a = 1./4.; b = 1./2.;

// if((0 < x) && (x <= a))
//   {
//     if((y >= -x + a + b) && (y <= x + (b - a) + 1./2.))
//       {
// 	return 1;
//       }
//   }
//  else if((a < x) && (x <= a + 1./4.))
//    {
//      if((y >= x + b - a) && (y <= -x + a + b + 1./2.))
//        {
// 	 return 1;
//        }
//    }

