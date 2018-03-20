/*
%function to compute the rank-one SFPCA solution
%L1 sparse penalties
%for fixed regularization params
%%%%%%%%%%%%%%%%%%%%%%%%
%inputs:
%x: n x p data matrix 
%lamu: sparsity parameter for u
%lamv: sparsity parameter for v
%alphau: smoothness parameter for u
%alphav: smoothness parameter for v
%Omegu: n x n positive semi-definite matrix for roughness penalty on u
%Omegv: p x p positive semi-definite matrix for roughness penalty on v
%startu: n  vector of starting values for U;  if startu=0, then
%algorithm initialized to the rank-one SVD solution
%startv: p vector of starting values for V;  if startv=0, then
%algorithm initialized to the rank-one SVD solution
%posu: non-negativity indicator - posu = True imposes non-negative
%constraints on u, posu = False otherwise
%posv: non-negativity indicator - posv = True imposes non-negative
%constraints on v, posv = False otherwise
%maxit: maximum number of alternating regressions steps
%%%%%%%%%%%%%%%%
%outputs:
%U: n x 1 left SFPC
%V p x 1 right SFPC
%d: the associated singular value
%Xhat: the deflated residual matrix
*/
#include <RcppEigen.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]

using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision


//proximal operator
VectorXd soft_thr(VectorXd a, double lam, bool pos){
	VectorXd temp,b,SignMat;
	if(!pos){
		//In MATLAB u = sign(a).*max(abs(a) - lam,0);
		b = a.array().abs();
		b.array() -= lam;
		b.array() = b.array().max(0);
		SignMat = a.cwiseSign();
		temp = (SignMat.array() * b.array()).matrix();
		return temp;
	}
	else{
		//In MATLAB u = max(a - lam,0);
		a.array() -= lam;
		temp.array() = a.array().max(0);
		return temp;
	}
}
// [[Rcpp::export]]
Rcpp::List sfpca_fixed(
	MatrixXd x,
	double lamu,
	double lamv,
	double alphau,
	double alphav,
	MatrixXd Omegu,
	MatrixXd Omegv,
	VectorXd startu,
	VectorXd startv,
	bool posu,
	bool posv,
	double maxit,
	)
{
	VectorXd u,v,oldu,oldv,oldui,oldvi,utild,vtild;
	double n,p,d;
	//Row size of data matrix
	n = x.rows();
	//Column size of data matrix
	p = x.cols();
	Eigen::SparseMatrix<double> Su(n,n) ,Sv(p,p);
  	Su.setIdentity();
	Sv.setIdentity();
	Su += n*alphau*Omegu.sparseView();
	Sv += p*alphav*Omegv.sparseView();
	//max eigen value of Su,added 0.1 to match with the matlab implmentation
	double Lu = MatrixXd(Su).eigenvalues().real().maxCoeff() + 0.1;
	//max eigen value of Sv,added 0.1 to match with the matlab implmentation
	double Lv = MatrixXd(Sv).eigenvalues().real().maxCoeff() + 0.1;
	//convergence threshold
	double thr = 1e-6;
	MatrixXd xhat = x;
	
	if(startu.sum()==0){
		//Divide and conquer SVD method
		Eigen::BDCSVD<MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
		MatrixXd utemp,vtemp;
		//calculatind svd of matrix x
		utemp = svd.matrixU();
		vtemp = svd.matrixV();
		//initialize u, v to SVD solution with max singular value
		u = utemp.col(0);
		v = vtemp.col(0);
		//initialize d with max singular value
		d = (svd.singularValues()).maxCoeff();
	}
	else{
		//initialize u, v as given by user
		u = startu;
		v = startv;
	}

	double indo=1,iter=0;
	//loop untill convergence and before maximum allowed iterations
	while((indo>thr) && (iter<maxit)){
		//store values of u and v before change
		oldu = u;
		oldv = v;
		double indu = 1;
		//loop untill convergence
		while(indu>thr){
			//store value of u
			oldui = u;
			//proximal operator of Pu() penalty 
			utild = u + (xhat*v - Su*u)/Lu;
			//updating u
			u = soft_thr(utild, lamu/Lu, posu);
			double unorm = u.norm() ;
			if(unorm>0){
				//Su norm of u
				double mod = ((u.transpose())*Su*u)(0,0);
				mod = std::sqrt(mod);
				u = u/mod;
			}
			else{
				u = VectorXd::Zero(n,1);
			}
			//fractional change in u
			indu = (u-oldui).norm()/oldui.norm() ;
		}

		double indv = 1;
		//loop untill convergence
		while(indv>thr){
			//store value of v
			oldvi = v;
			//proximal operator of Pv() penalty
			vtild = v + ((xhat.transpose())*u - Sv*v)/Lv;
			//updating v
			v = soft_thr(vtild, lamv/Lv, posv);
			double vnorm = v.norm();
			if(vnorm>0){
				//Sv norm of v
				double mod = ((v.transpose())*Sv*v)(0,0);
				mod = std::sqrt(mod);
				v = v/mod;
			}
			else{
				v = VectorXd::Zero(p,1);
			}
			//fractional change in v
			indv = (v-oldvi).norm()/oldvi.norm();
		}
		//total fractional change in u and v
		indo = (oldu-u).norm()/oldu.norm() + (oldv-v).norm()/oldv.norm();
		iter++;
	}
	//prepare final return forms
	u = u/u.norm();
	v = v/v.norm();
	d = (((u).transpose())*xhat*(v))(0,0);
	xhat = xhat - (d)*(u)*((v).transpose());
	// return list
	return Rcpp::List::create(Rcpp::Named("U") = u,
	                     Rcpp::Named("V") = v,
	                     Rcpp::Named("D") = d,
	                     Rcpp::Named("Xhat") = xhat);
}
