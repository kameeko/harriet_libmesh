#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"
#include "libmesh/fem_context.h"
#include "libmesh/equation_systems.h"
#include "libmesh/point_locator_tree.h"

#include <numeric>

using namespace libMesh;

class ConvDiff_MprimeSys : public FEMSystem
{
public:

  // Constructor
  ConvDiff_MprimeSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){

    qoi.resize(1);

    GetPot infile("convdiff_mprime.in");
    std::string find_velocity_here = infile("velocity_file","velsTtrim.txt");
    std::string find_data_here = infile("data_file","Measurements_top6.dat");
    qoi_option = infile("QoI_option",1);
    
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
		
		sadj_auxc_point_stash.resize(datavals.size()); //DEBUG
		sadj_c_point_stash.resize(datavals.size()); //DEBUG
		
		//find elements in which data points reside
	  PointLocatorTree point_locator(this->get_mesh());
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	Point data_point = datapts[dnum];
	  	Elem *this_elem = const_cast<Elem *>(point_locator(data_point));
	  	dataelems.push_back(this_elem->id());
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
  //virtual void postprocess();
  void postprocess(unsigned int dbg_step = 0); //DEBUG
  
  //to calculate QoI
  virtual void element_postprocess(DiffContext &context);
  
  //for adjoint
  virtual void element_qoi_derivative(DiffContext &context, const QoISet & qois);
  
  //get cell's constribution to M_HF(psiLF) term of QoI error estimate
  double get_MHF_psiLF(int elem_ind){
  	return MHF_psiLF[elem_ind];
  }
  double get_MHF_psiLF(){
  	return std::accumulate(MHF_psiLF.begin(),MHF_psiLF.end(),0.0);
  }
  
  //get cell's constribution to M_LF(psiLF) term of QoI error estimate
  double get_MLF_psiLF(int elem_ind){
  	return MLF_psiLF[elem_ind];
  }
  double get_MLF_psiLF(){
  	return std::accumulate(MLF_psiLF.begin(),MLF_psiLF.end(),0.0);
  }
  
  //DEBUGGING
  double get_half_adj_weighted_resid(int elem_ind){
  	return half_sadj_resid[elem_ind];
  }
  double get_half_adj_weighted_resid(){
  	return std::accumulate(half_sadj_resid.begin(), half_sadj_resid.end(), 0.0);
  }

  // Indices for each variable;
  unsigned int c_var, zc_var, fc_var, aux_c_var, aux_zc_var, aux_fc_var;
  
  Real beta_grad, beta_mag; //regularization parameter
  Real k; //diffusion coefficient
  Real R; //reaction coefficient
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  std::vector<dof_id_type> dataelems;
  
  //velocity field
	std::vector<Real> x_pts;
	std::vector<std::vector<Real> > y_pts;
	std::vector<std::vector<NumberVectorValue> > vel_field;
	
	//M_HF(psiLF) and M_LF(psiLF) terms of QoI error estimate
	std::vector<Real> MHF_psiLF;
	std::vector<Real> MLF_psiLF;
	
	//DEBUGGING
	std::vector<Real> half_sadj_resid;
	std::vector<std::vector<Real> > sadj_c_stash; std::vector<std::vector<Gradient> > sadj_gradc_stash;
	std::vector<std::vector<Real> > sadj_zc_stash; std::vector<std::vector<Gradient> > sadj_gradzc_stash;
	std::vector<std::vector<Real> > sadj_fc_stash; std::vector<std::vector<Gradient> > sadj_gradfc_stash;
	std::vector<std::vector<Real> > sadj_auxc_stash; std::vector<std::vector<Gradient> > sadj_gradauxc_stash;
	std::vector<std::vector<Real> > sadj_auxzc_stash; std::vector<std::vector<Gradient> > sadj_gradauxzc_stash;
	std::vector<std::vector<Real> > sadj_auxfc_stash; std::vector<std::vector<Gradient> > sadj_gradauxfc_stash;
	std::vector<Real> sadj_auxc_point_stash; std::vector<Real> sadj_c_point_stash;
	unsigned int debug_step; //if 1, fill up sadj stash; if 2, calculate half_sadj_resid

  //options for QoI location and nature
  int qoi_option;
};
