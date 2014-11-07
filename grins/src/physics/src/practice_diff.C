// This class
#include "grins/practice_diff.h"

// GRINS
#include "grins/assembly_context.h"
#include "grins/generic_ic_handler.h"
#include "grins/grins_physics_names.h" 
#include "grins/practice_bc_handling.h"
#include "grins/grins_enums.h"

// libMesh
#include "libmesh/quadrature.h"
#include "libmesh/fem_system.h"
#include "libmesh/string_to_enum.h"

namespace GRINS{

	PracticeDiff::PracticeDiff( const GRINS::PhysicsName& physics_name, const GetPot& input )
		: Physics(physics_name,input), 
			_k(input("Physics/PracticeDiffusion/k", 1.0)),
			_fefamily( libMesh::Utility::string_to_enum<GRINSEnums::FEFamily>( 
					input("Physics/"+physics_name+"/fe_family", "LAGRANGE"))){
		
		this->_bc_handler = new PracticeBCHandling( physics_name, input );
    this->_ic_handler = new GenericICHandler( physics_name, input );
    
		return;
	}
	
	PracticeDiff::~PracticeDiff(){ return; }
	
	void PracticeDiff::init_variables( libMesh::FEMSystem* system ){
		//polynomial order and finite element type for concentration variable
		unsigned int conc_p = 1;
		
		_c_var = system->add_variable("c", static_cast<GRINSEnums::Order>(conc_p), _fefamily); 
	}
	
	void PracticeDiff::init_context( AssemblyContext& context){
		context.get_element_fe(_c_var)->get_JxW();
    context.get_element_fe(_c_var)->get_phi();
    context.get_element_fe(_c_var)->get_dphi();
    context.get_element_fe(_c_var)->get_xyz();

    context.get_side_fe(_c_var)->get_JxW();
    context.get_side_fe(_c_var)->get_phi();
    context.get_side_fe(_c_var)->get_dphi();
    context.get_side_fe(_c_var)->get_xyz();

    return;
	}
	
	void PracticeDiff::element_time_derivative( bool compute_jacobian,
						AssemblyContext& context,
						CachedValues& /*cache*/ ){
	
		// The number of local degrees of freedom in each variable.
    const unsigned int n_c_dofs = context.get_dof_indices(_c_var).size();

    // We get some references to cell-specific data that
    // will be used to assemble the linear system.

    // Element Jacobian * quadrature weights for interior integration.
    const std::vector<libMesh::Real> &JxW =
      context.get_element_fe(_c_var)->get_JxW();

    // The temperature shape function gradients (in global coords.)
    // at interior quadrature points.
    const std::vector<std::vector<libMesh::RealGradient> >& dphi =
      context.get_element_fe(_c_var)->get_dphi();
    const std::vector<std::vector<libMesh::Real> >& phi = context.get_element_fe(_c_var)->get_phi();

    const std::vector<libMesh::Point>& q_points = 
      context.get_element_fe(_c_var)->get_xyz();

    libMesh::DenseSubMatrix<libMesh::Number> &Kcc = context.get_elem_jacobian(_c_var, _c_var); 

    libMesh::DenseSubVector<libMesh::Number> &Fc = context.get_elem_residual(_c_var);

    // Now we will build the element Jacobian and residual.
    // Constructing the residual requires the solution and its
    // gradient from the previous timestep.  This must be
    // calculated at each quadrature point by summing the
    // solution degree-of-freedom values by the appropriate
    // weight functions.
    unsigned int n_qpoints = context.get_element_qrule().n_points();

    for (unsigned int qp=0; qp != n_qpoints; qp++){

			libMesh::Gradient grad_c;
			grad_c = context.interior_gradient(_c_var, qp);

			const libMesh::Real f = this->forcing( q_points[qp] );
	
			// First, an i-loop over the  degrees of freedom.
			for (unsigned int i=0; i != n_c_dofs; i++){
				Fc(i) += JxW[qp] *(_k*(dphi[i][qp]*grad_c) - f*phi[i][qp]);  // diffusion term

				if (compute_jacobian){
					for (unsigned int j=0; j != n_c_dofs; j++){
						Kcc(i,j) += JxW[qp]*( _k*(dphi[i][qp]*dphi[j][qp]) ); // diffusion term
					} // end of the inner dof (j) loop
			  } // end - if (compute_jacobian && context.get_elem_solution_derivative())

			} // end of the outer dof (i) loop
    } // end of the quadrature point (qp) loop

    return;
	
	}
	
	inline
	libMesh::Real PracticeDiff::forcing( const libMesh::Point& p){
		return exp(-10*(pow(p(0)-0.25,2)+pow(p(1)-0.25,2)));
	}

} //namespace GRINS
