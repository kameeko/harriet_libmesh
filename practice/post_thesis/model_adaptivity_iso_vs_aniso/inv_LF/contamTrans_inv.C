#include "libmesh/getpot.h"

#include "contamTrans_inv.h"

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
void ContamTransSysInv::init_data(){
  const unsigned int dim = this->get_mesh().mesh_dimension();

  GetPot infile("contamTrans.in");
  
  //regularization
	beta = infile("beta", 0.1);
	
  unsigned int poly_order = infile("poly_order",1);
  std::string fefamily = infile("fe_family", std::string("LAGRANGE"));
  c_var = this->add_variable("c", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));
  f_var = this->add_variable("f", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));
  z_var = this->add_variable("z", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));

  //indicate variables that change in time
  this->time_evolving(c_var);
  this->time_evolving(f_var);
  this->time_evolving(z_var);

  // Useful debugging options'
	// Set verify_analytic_jacobians to positive to use
	this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 0.);
	this->print_jacobians = infile("print_jacobians", false);
	this->print_element_jacobians = infile("print_element_jacobians", false);

	//set Dirichlet boundary conditions (none in this case)

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

  useSUPG = infile("use_stabilization",false);
  stab_opt = infile("stabilization_option",1);

	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

//context initialization
void ContamTransSysInv::init_context(DiffContext & context){
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* c_elem_fe;
  FEBase* c_side_fe;

  ctxt.get_element_fe(c_var, c_elem_fe);
	ctxt.get_side_fe(c_var, c_side_fe );

	c_elem_fe->get_JxW();
	c_elem_fe->get_phi();
	c_elem_fe->get_dphi();
	c_elem_fe->get_d2phi();
	c_elem_fe->get_xyz();

	c_side_fe->get_JxW();
	c_side_fe->get_phi();
	c_side_fe->get_dphi();
	c_side_fe->get_xyz();
}

//element residual and jacobian calculations
bool ContamTransSysInv::element_time_derivative(bool request_jacobian, DiffContext & context)
{
  const unsigned int dim = this->get_mesh().mesh_dimension();

  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* c_elem_fe = NULL;
  ctxt.get_element_fe( c_var, c_elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = c_elem_fe->get_JxW();

  const std::vector<std::vector<Real> >& phi = c_elem_fe->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = c_elem_fe->get_dphi();
  const std::vector<std::vector<RealTensor> >& d2phi = c_elem_fe->get_d2phi();

  // Physical location of the quadrature points
  const std::vector<Point>& qpoint = c_elem_fe->get_xyz();

  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices( c_var ).size();

  // The subvectors and submatrices we need to fill:
	DenseSubMatrix<Number> &J_c_z = ctxt.get_elem_jacobian(c_var, z_var);
	DenseSubMatrix<Number> &J_c_c = ctxt.get_elem_jacobian(c_var, c_var);

	DenseSubMatrix<Number> &J_z_c = ctxt.get_elem_jacobian(z_var, c_var);
	DenseSubMatrix<Number> &J_z_f = ctxt.get_elem_jacobian(z_var, f_var);

	DenseSubMatrix<Number> &J_f_z = ctxt.get_elem_jacobian(f_var, z_var);
	DenseSubMatrix<Number> &J_f_f = ctxt.get_elem_jacobian(f_var, f_var);
	
	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( c_var );
	DenseSubVector<Number> &Rz = ctxt.get_elem_residual( z_var );
	DenseSubVector<Number> &Rf = ctxt.get_elem_residual( f_var );

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
      c = ctxt.interior_value(c_var, qp),
      z = ctxt.interior_value(z_var, qp),
      f = ctxt.interior_value(f_var, qp);
    Gradient 
      grad_c = ctxt.interior_gradient(c_var, qp),
      grad_z = ctxt.interior_gradient(z_var, qp),
      grad_f = ctxt.interior_gradient(f_var, qp);
    NumberTensorValue hess_c = ctxt.interior_hessian(c_var, qp); //for SUPG

    //velocity vector
    NumberVectorValue U(vx, 0.0, 0.0);
    
    //SUPG
    double tau = 0.0;
    if(useSUPG){ //assuming isotropic dispersion for now
      if(stab_opt == 1){
        //version 1, copied from MILO code
        double C1 = 4.0;
        double C2 = 2.0;
        double k = dispTens(0,0)*porosity;
        if(dispTens(0,0) != dispTens(1,1) || dispTens(1,1) != dispTens(2,2))
          std::cout << "SUPG currently assumed isotropic dispersion..." << std::endl;
        Real h = ctxt.get_elem().hmax();
        tau = 1./((C1*k)/(h*h) + (C2*sqrt(U*U)/h));
      }else if(stab_opt == 2){
        //version 2, from http://ta.twi.tudelft.nl/TWA_Reports/06/06-03.pdf
        double k = dispTens(0,0)*porosity;
        if(dispTens(0,0) != dispTens(1,1) || dispTens(1,1) != dispTens(2,2))
          std::cout << "SUPG currently assumed isotropic dispersion..." << std::endl;
        Real h = ctxt.get_elem().hmax();
        Real Pe = sqrt(U*U)*h/(2.*k); //element Peclet number
        tau = (h/(2.*sqrt(U*U)))*(1./tanh(Pe) + 1./Pe);
      }else if(stab_opt == 3){
        //version 3, from Becker + Braack (2002), assuming linear basis functions for now
        double C = 12.0; //assuming linear basis functions
        double delta0 = 0.4; //suggested 0.2 <= delta_0 <= 0.5 for linear basis functions
        Real hK = ctxt.get_elem().hmax();
        Real betaK = std::max(std::max(std::abs(U(0)),std::abs(U(1))),std::abs(U(2)));
        double k = dispTens(0,0)*porosity;
        if(dispTens(0,0) != dispTens(1,1) || dispTens(1,1) != dispTens(2,2))
          std::cout << "SUPG currently assumed isotropic dispersion..." << std::endl;
        tau = delta0*hK*hK/(C*k + betaK*hK);
      }else if(stab_opt == 4){
        //version 4, from http://www.scielo.br/pdf/jbsmse/v32n3/v32n3a13.pdf
        //porosity in this paper is only attached to time derivative...not quite the same as in Ewing+Weeks...
        Real h_e = ctxt.get_elem().hmax();
        //double Pe_e = 0.5*h_e*(pow(sqrt(U*U),3.)/(U*(dispTens*U))); //as in paper
        //tau = 0.5*(h_e/sqrt(U*U))*std::min(Pe_e/3.,1.); //as in paper
        double Pe_e = 0.5*h_e*(pow(sqrt(U*U),3.)/(U*(porosity*dispTens*U))); //adapted to our equation?
        tau = 0.5*(h_e/sqrt(U*U))*std::min(Pe_e/3.,1.); //adapted to our equation?
      }else{
        std::cout << "Invalid stabilization option. No stabilization used." << std::endl;
      }
    }

    // First, an i-loop over the  degrees of freedom.
    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      //dL/dz = 0
      Rz(i) += JxW[qp]*
                 (-(dispTens*(porosity*grad_c))*dphi[i][qp] // Dispersion Term
		             - (U*grad_c)*phi[i][qp] // Convection Term
		             - (react_rate*(porosity*c))*phi[i][qp] // Reaction Term
		             + f*phi[i][qp]); // Source term
		       
		  //dL/dc = 0
		  Rc(i) += JxW[qp]*
		            (-(dispTens*(porosity*grad_z))*dphi[i][qp] // Dispersion Term
		             + (U*grad_z)*phi[i][qp] // Convection Term
		             - (react_rate*(porosity*z))*phi[i][qp]); // Reaction Term
		  
		  //dL/df = 0
		  Rf(i) += JxW[qp]*(beta*grad_f*dphi[i][qp] + z*phi[i][qp]); 
		       
		  if(useSUPG){
		    //dL/dz = 0
		    Rz(i) += JxW[qp]*((tau*U*dphi[i][qp])*
		               (porosity*dispTens.contract(hess_c) //Dispersion term
		               - (U*grad_c) // Convection Term
		               - (react_rate*(porosity*c)) // Reaction Term
		               + f)); // Source term
		    
		    //dL/dc = 0
		    Rc(i) += JxW[qp]*((tau*U*grad_z)*
		               (porosity*dispTens.contract(d2phi[i][qp]) //Dispersion term
		               - (U*dphi[i][qp]) // Convection Term
		               - (react_rate*(porosity*phi[i][qp])))); // Reaction Term
		    
		    //dL/df = 0
		    Rf(i) += JxW[qp]*(tau*U*grad_z)*phi[i][qp];
      }
      
      if (request_jacobian && ctxt.get_elem_solution_derivative())
      {
	      for (unsigned int j=0; j != n_c_dofs; j++)
	      {
	        J_z_c(i,j) += JxW[qp]*
	                       ((-dispTens*(porosity*dphi[j][qp]))*dphi[i][qp] // Dispersion
			                   - (U*dphi[j][qp])*phi[i][qp] // Convection
			                   - (react_rate*(porosity*phi[j][qp]))*phi[i][qp]); // Reaction Term
			    J_z_f(i,j) += JxW[qp]*phi[j][qp]*phi[i][qp];
			    
			    J_c_z(i,j) += JxW[qp]*
			                    (-(dispTens*(porosity*dphi[j][qp]))*dphi[i][qp] // Dispersion Term
		                       + (U*dphi[j][qp])*phi[i][qp] // Convection Term
		                       - (react_rate*(porosity*phi[j][qp]))*phi[i][qp]); // Reaction Term
			    
			    J_f_z(i,j) += JxW[qp]*(phi[j][qp]*phi[i][qp]); 
			    J_f_f(i,j) += JxW[qp]*(beta*dphi[j][qp]*dphi[i][qp]); 
			    if(useSUPG){
			      J_z_c(i,j) += JxW[qp]*((tau*U*dphi[i][qp])*
		                 (porosity*dispTens.contract(d2phi[j][qp]) //Dispersion term
		                 - (U*dphi[j][qp]) // Convection Term
		                 - (react_rate*(porosity*phi[j][qp])))); // Reaction Term
		        J_z_f(i,j) += JxW[qp]*((tau*U*dphi[i][qp])*phi[j][qp]);
		        
			      J_c_z(i,j) += JxW[qp]*((tau*U*dphi[j][qp])*
		                       (porosity*dispTens.contract(d2phi[i][qp]) //Dispersion term
		                       - (U*dphi[i][qp]) // Convection Term
		                       - (react_rate*(porosity*phi[i][qp])))); // Reaction Term
			      
			      J_f_z(i,j) += JxW[qp]*(tau*U*dphi[j][qp])*phi[i][qp];
          }
	      } // end of the inner dof (j) loop
      } // end - if (request_jacobian && context.get_elem_solution_derivative())

    } // end of the outer dof (i) loop
  } // end of the quadrature point (qp) loop

  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
  	Point data_point = datapts[dnum];
  	if(ctxt.get_elem().contains_point(data_point) && (accounted_for[dnum]>=ctxt.get_elem().id()) ){
  	
  		//help avoid double-counting if data from edge of elements, but may mess with jacobian check
  		accounted_for[dnum] = ctxt.get_elem().id(); 
  		
  		Number cpred = ctxt.point_value(c_var, data_point);
  		Number cstar = datavals[dnum];

  		unsigned int dim = ctxt.get_system().get_mesh().mesh_dimension();
	    FEType fe_type = ctxt.get_element_fe(c_var)->get_fe_type();
	    
	    //go between physical and reference element
	    Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
	    
      std::vector<Real> point_phi(n_c_dofs);
    	for (unsigned int i=0; i != n_c_dofs; i++){
    		//get value of basis function at mapped point in reference (master) element
        point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
      }
      
      for (unsigned int i=0; i != n_c_dofs; i++){
	  		Rc(i) += -(cpred - cstar)*point_phi[i];
  
				if (request_jacobian){
					for (unsigned int j=0; j != n_c_dofs; j++)
						J_c_c(i,j) += -point_phi[j]*point_phi[i] ;
			  }
  
			}
  	}
  }

  return request_jacobian;
}

//for non-Dirichlet boundary conditions and the bit from diffusion term
bool ContamTransSysInv::side_time_derivative(bool request_jacobian, DiffContext & context)
{

  const unsigned int dim = this->get_mesh().mesh_dimension();

  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEBase* side_fe = NULL;
  ctxt.get_side_fe(c_var, side_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = side_fe->get_JxW();

  // Side basis functions
  const std::vector<std::vector<Real> > &phi = side_fe->get_phi();

  // Side Quadrature points
  const std::vector<Point > &qside_point = side_fe->get_xyz();

  //normal vector
  const std::vector<Point> &face_normals = side_fe->get_normals();

  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices(c_var).size();

  // The subvectors and submatrices we need to fill:
  DenseSubMatrix<Number> &J_c_z = ctxt.get_elem_jacobian(c_var, z_var);

	DenseSubMatrix<Number> &J_z_c = ctxt.get_elem_jacobian(z_var, c_var);
	
	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( c_var );
	DenseSubVector<Number> &Rz = ctxt.get_elem_residual( z_var );
	//Rf gets no contribution from sides


  unsigned int n_qpoints = ctxt.get_side_qrule().n_points();

  bool isWest = false;
  bool isEast = false;
  if (dim == 3){
    isWest = ctxt.has_side_boundary_id(4);
    isEast = ctxt.has_side_boundary_id(2);
  }
  else if (dim == 2){
    isWest = ctxt.has_side_boundary_id(3);
    isEast = ctxt.has_side_boundary_id(1);
  }
  
  //set (in)flux boundary condition on west side
  //homogeneous neumann (Danckwerts) outflow boundary condition on east side
  //no-flux (equivalently, homoegenous neumann) boundary conditions on north, south, top, bottom sides
  //"strong" enforcement of boundary conditions
  for (unsigned int qp=0; qp != n_qpoints; qp++)
  {
    Number c = ctxt.side_value(c_var, qp),
           z = ctxt.side_value(z_var, qp);

    //velocity vector
    NumberVectorValue U(vx, 0.0, 0.0);

    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      if(isWest) //west boundary
        Rz(i) += JxW[qp]*(U*face_normals[qp]*c - bsource*vx)*phi[i][qp];
      if(isEast)
        Rc(i) += JxW[qp]*(-U*face_normals[qp]*z)*phi[i][qp];
      
      if(request_jacobian && context.get_elem_solution_derivative())
      {
        for (unsigned int j=0; j != n_c_dofs; j++)
	      {
          if(isWest)
            J_z_c(i,j) += JxW[qp]*(U*face_normals[qp]*phi[j][qp])*phi[i][qp];
          if(isEast)
            J_c_z(i,i) += JxW[qp]*(-U*face_normals[qp]*phi[j][qp])*phi[i][qp];
	      }
      } // end - if (request_jacobian && context.get_elem_solution_derivative())
    } //end of outer dof (i) loop
  }

  return request_jacobian;
}

//generate data
void ContamTransSysInv::postprocess(){

	//reset computed QoIs
  computed_QoI[0] = 0.0;

  FEMSystem::postprocess();

  this->comm().sum(computed_QoI[0]);
  
}
