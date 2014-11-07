#ifndef GRINS_PRAC_CDR_INV_H
#define GRINS_PRAC_CDR_INV_H

// GRINS
#include "grins/grins_enums.h"
#include "grins/physics.h"

//libMesh
#include "libmesh/enum_order.h"
#include "libmesh/enum_fe_family.h"

namespace GRINS{

	class PracticeCDRinv : public Physics{
	
	public:
	
		PracticeCDRinv( const GRINS::PhysicsName& physics_name, const GetPot& input);
		~PracticeCDRinv();
		
		//! Initialize variables for this physics.
    virtual void init_variables( libMesh::FEMSystem* system );

    //! Initialize context for added physics variables
    virtual void init_context( AssemblyContext& context );
    
    //! Time dependent part(s) of physics for element interiors
    virtual void element_time_derivative( bool compute_jacobian,
					  AssemblyContext& context,
					  CachedValues& cache );
	
	protected:
		
		libMesh::Number _k; //diffusion coefficient
		libMesh::Real _R; //reaction coefficient
		libMesh::Real _beta; //regularization coefficient
		
		VariableIndex _c_var; //index for concentration
		VariableIndex _zc_var; //index for adjoint
		VariableIndex _fc_var; //index for parameter
		
		GRINSEnums::FEFamily _fefamily; //element type
	
	private:
	
		PracticeCDRinv(); //not sure what the point of this is, but the others have it...
		
	  //velocity field
		std::vector<libMesh::Real> x_pts;
		std::vector<std::vector<libMesh::Real> > y_pts;
		std::vector<std::vector<libMesh::NumberVectorValue> > vel_field;
		
		//data
		std::vector<libMesh::Point> datapts; 
  	std::vector<libMesh::Real> datavals;

	}; //end class

} //end namespace

#endif //GRINS_PRAC_CDR_INV_H
