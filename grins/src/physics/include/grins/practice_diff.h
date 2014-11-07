#ifndef GRINS_PRAC_DIFF_H
#define GRINS_PRAC_DIFF_H

// GRINS
#include "grins/grins_enums.h"
#include "grins/physics.h"

//libMesh
#include "libmesh/enum_order.h"
#include "libmesh/enum_fe_family.h"

namespace GRINS{

	class PracticeDiff : public Physics{
	
	public:
	
		PracticeDiff( const GRINS::PhysicsName& physics_name, const GetPot& input);
		~PracticeDiff();
		
		//! Initialize variables for this physics.
    virtual void init_variables( libMesh::FEMSystem* system );

    //! Initialize context for added physics variables
    virtual void init_context( AssemblyContext& context );
    
    //! Time dependent part(s) of physics for element interiors
    virtual void element_time_derivative( bool compute_jacobian,
					  AssemblyContext& context,
					  CachedValues& cache );
	
	protected:
	
		libMesh::Real forcing( const libMesh::Point& p);
		
		libMesh::Number _k; //diffusion coefficient
		VariableIndex _c_var; //index for concentration
		
		GRINSEnums::FEFamily _fefamily; //element type
	
	private:
	
		PracticeDiff(); //not sure what the point of this is, but the others have it...

	}; //end class

} //end namespace

#endif //GRINS_PRAC_DIFF_H
