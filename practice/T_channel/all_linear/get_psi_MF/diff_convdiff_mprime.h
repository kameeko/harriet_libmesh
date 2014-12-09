#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"

using namespace libMesh;

class Diff_ConvDiff_MprimeSys : public FEMSystem
{
public:

  // Constructor
  Diff_ConvDiff_MprimeSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){
    
    GetPot infile("diff_convdiff_mprime.in");
		std::string find_velocity_here = infile("velocity_file","velsTtrim.txt");
		std::string find_data_here = infile("data_file","Measurements_top6.dat");
    
    if(FILE *fp=fopen(find_velocity_here.c_str(),"r")){
  		Real u, v, x, y;
  		Real prevx = 1.e10;
  		std::vector<Real> tempvecy;
  		std::vector<NumberVectorValue> tempvecvel;
  		int flag = 1;
  		while(flag != -1){
  			flag = fscanf(fp, "%lf %lf %lf %lf",&u,&v,&x,&y);
  			if(flag != -1){
  				if(x != prevx){
  					x_pts.push_back(x);
  					prevx = x;
  					if(x_pts.size() > 1){
  						y_pts.push_back(tempvecy);
  						vel_field.push_back(tempvecvel);
						}
  					tempvecy.clear(); 
  					tempvecvel.clear();
  					tempvecy.push_back(y); 
  					tempvecvel.push_back(NumberVectorValue(u,v));
  				}
  				else{
  					tempvecy.push_back(y); 
  					tempvecvel.push_back(NumberVectorValue(u,v));
  				}
  			}
  		}
  		y_pts.push_back(tempvecy);
  		vel_field.push_back(tempvecvel);
  	}
		if(FILE *fp=fopen(find_data_here.c_str(),"r")){
	  	Real x, y, value;
	  	int flag = 1;
	  	while(flag != -1){
	  		flag = fscanf(fp,"%lf %lf %lf",&x,&y,&value);
	  		if(flag != -1){
					datapts.push_back(Point(x,y));
					datavals.push_back(value);
	  		}
	  	}
	  	fclose(fp);
	  }
  }

  // System initialization
  virtual void init_data ();

  // Context initialization
  virtual void init_context(DiffContext &context);

  // Element residual and jacobian calculations
  // Time dependent parts
  virtual bool element_time_derivative (bool request_jacobian,
                                        DiffContext& context);

  // Postprocessed output
  virtual void postprocess();
  
  //DEBUG
  virtual void element_postprocess(DiffContext &context);
  double get_MHF_psiLF(int elem_ind){
  	return MHF_psiLF[elem_ind];
  }
  double get_MHF_psiLF(){
  	return std::accumulate(MHF_psiLF.begin(),MHF_psiLF.end(),0.0);
  }
  double get_MLF_psiLF(int elem_ind){
  	return MLF_psiLF[elem_ind];
  }
  double get_MLF_psiLF(){
  	return std::accumulate(MLF_psiLF.begin(),MLF_psiLF.end(),0.0);
  }

  // Indices for each variable;
  unsigned int c_var, zc_var, fc_var, aux_c_var, aux_zc_var, aux_fc_var;
  
  Real beta; //regularization parameter
  Real k; //diffusion coefficient
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  
  //velocity field
	std::vector<Real> x_pts;
	std::vector<std::vector<Real> > y_pts;
	std::vector<std::vector<NumberVectorValue> > vel_field;
	
	int diff_subdomain_id, convdiff_subdomain_id;

	//DEBUG
	std::vector<Real> MHF_psiLF;
	std::vector<Real> MLF_psiLF;

  // Returns the value of a forcing function at point p.  This value
  // depends on which application is being used.
  Point forcing(const Point& p);
};
